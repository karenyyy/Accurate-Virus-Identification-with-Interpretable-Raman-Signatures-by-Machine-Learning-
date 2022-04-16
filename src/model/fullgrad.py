from src.config import *

class FullGradExtractor:
    # Extract tensors needed for FullGrad using hooks

    def __init__(self, model, im_size=(1, 1196)):
        self.model = model
        self.im_size = im_size

        self.biases = []
        self.feature_grads = []
        self.grad_handles = []

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):

                # Register feature-gradient hooks for each layer
                handle_g = m.register_backward_hook(self._extract_layer_grads)
                self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)

                '''
                b: 
                    torch.Size([128])
                    torch.Size([64])
                    torch.Size([32])
                    torch.Size([16])
                    torch.Size([3])
                '''
                if (b is not None): self.biases.append(b)

    def _extract_layer_bias(self, module):
        # extract bias of each layer

        # for batchnorm, the overall "bias" is different
        # from batchnorm bias parameter.
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b)
        # Thus (-m*w/s + b) is the effective bias of batchnorm

        if isinstance(module, nn.BatchNorm1d):
            b = - (module.running_mean * module.weight
                   / torch.sqrt(module.running_var + module.eps)) + module.bias
            return b.data
        elif module.bias is None:
            return None
        else:
            return module.bias.data

    def getBiases(self):
        # dummy function to get biases
        return self.biases

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            self.feature_grads.append(out_grad[0])

    def getFeatureGrads(self, x, output_scalar):

        # Empty feature grads list
        self.feature_grads = []
        '''
                       torch.Size([1, 3]) 
                       torch.Size([1, 16, 148]) 
                       torch.Size([1, 32, 298]) 
                       torch.Size([1, 64, 597])
                       torch.Size([1, 128, 1196])
        '''

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs=output_scalar, inputs=x)[0]

        return input_gradients, self.feature_grads


class FullGrad():

    def __init__(self, model, im_size=(1, 1196)):
        self.model = model
        self.im_size = (1,) + im_size
        self.model_ext = FullGradExtractor(model, im_size)
        self.biases = self.model_ext.getBiases()
        self.checkCompleteness()

    def checkCompleteness(self):
        """
        Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases of non-linearities).

        """
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")

        # Random input image
        input = torch.randn(self.im_size).to(device)

        # Get raw outputs
        self.model.eval()

        raw_output = self.model(input)

        # Compute full-gradients and add them up
        input_grad, bias_grad = self.fullGradientDecompose(input, target_class=None)

        fullgradient_sum = (input_grad * input).sum()
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum()

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients."
        err_string = "Completeness test failed! Raw output = " + str(
            raw_output.max().item()) + " Full-gradient sum = " + str(fullgradient_sum.item())
        # assert isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=1e-4), err_string + err_message

    def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """

        self.model.eval()
        image = image.requires_grad_()

        out = self.model(image)

        loss = nn.CrossEntropyLoss()

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        if len(target_class.size()) > 1:
            target_class = target_class.squeeze(1)
        # Select the output unit corresponding to the target class
        # -1 compensates for negation in nll_loss function
        # output_scalar = loss(out, target_class)
        # loss(input, class) = -input[class]
        output_scalar = -1 * F.nll_loss(F.softmax(out), target_class, reduction='sum')

        #      -1 * F.nll_loss(F.softmax(out), target_class, reduction='sum'))
        input_gradient, feature_gradients = self.model_ext.getFeatureGrads(image, output_scalar)

        # Compute feature-gradients \times bias
        bias_times_gradients = []
        L = len(self.biases)

        for i in range(L):
            # feature gradients are indexed backwards
            # because of backprop

            g = feature_gradients[L - 1 - i]
            # reshape bias dimensionality to match gradients
            bias_size = [1] * len(g.size())
            bias_size[1] = self.biases[i].size(0)
            b = self.biases[i].view(tuple(bias_size))
            o = g * b.expand_as(g)

            bias_times_gradients.append(o)
        return input_gradient, bias_times_gradients

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input.view((input.size(0), -1))
        temp, _ = flatin.min(1, keepdim=True)
        input = input - temp.unsqueeze(1).unsqueeze(1)

        flatin = input.view((input.size(0), -1))
        temp, _ = flatin.max(1, keepdim=True)
        input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input

    def saliency(self, image, target_class=None):
        # FullGrad saliency

        self.model.eval()
        target_class = torch.tensor(target_class).cuda().unsqueeze(0)
        input_grad, bias_grad = self.fullGradientDecompose(image, target_class=target_class)

        # Input-gradient * image
        grd = input_grad * image

        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        im_size = image.size()

        # Aggregate Bias-gradients
        bias_cam = {}
        for i in range(len(bias_grad)):

            # Select only Conv layers
            if len(bias_grad[i].size()) == len(im_size):
                temp = self._postProcess(bias_grad[i])

                gradient = F.interpolate(temp, size=(im_size[1], im_size[2]),
                                         mode='bilinear', align_corners=True)

                cam += gradient.sum(1, keepdim=True)
                bias_cam[i] = gradient.sum(1, keepdim=True).data.cpu().numpy()

        return cam.data.cpu().numpy(), bias_cam
