function []=preprocessor_publish(filename,labelcol,front)
%% Preprocess the spectrum by spike filter, filter Savitzky–Golay filter and background subtraction
    %input: ./filename.csv of the spectrum (The csv is with wavenumber headers)
    %       labelcol(int): The number of columns that is label
    %       front(bool): true if the label is in front
    %output ./filename_preprocessed.csv with same format as input
    %       ***the preprocessed spectrum are interger for saving memory
    %
    %input sample: spectrum.csv
    % |sample|600|601|602|...
    % |  1   |10 |15 |11 |...
    % |  1   |12 |14 |12 |...
    %preprocessor_publish('spectrum',1,True)
    
    % Read csv to table
    csvtable=readtable(strcat(filename,'.csv'));
    
    % Get the spectrum
    if front
        rawsample=table2array(csvtable(:,labelcol+1:end));
    else
        rawsample=table2array(csvtable(:,1:end-labelcol));
    end
    
    % Get wavenumber
    x=rawsample(1,1:end);
    rawsample=rawsample(2:size(rawsample,1),:);
    
    % Get label (no usage in preprocess, only for output)
    if front
        label=csvtable(:,1:labelcol);
    else
        label=csvtable(:,end-labelcol+1:end);
    end
    
    sample=rawsample;
    % Apply spike filter
    sample=medfilt1(sample,10,[],2,'truncate');
    % Apply Savitzky–Golay filter
    sample=sgolayfilt(sample,5,21,ones(1,21),2);
    % Apply base linecorrection with msbackadj
    sample = msbackadj((1:size(sample,2))',sample(:,:)');
    sample = sample';

    % write to output in same format with preprocessed data
    % ***output int for saving Memory***
    if front
        writetable([label,array2table([x(1:end);int64(sample)],'VariableNames',compose(strcat('Var', '%d'), labelcol+1:size(x,2)+labelcol))],strcat(filename,'_preprocessed','.csv'),'WriteVariableNames',false);
    else
        writetable([array2table([x(1:end);int64(sample)]),label],strcat(filename,'_preprocessed','.csv'),'WriteVariableNames',false);
    end
    
    % play music after done
    beep on; beep;
end