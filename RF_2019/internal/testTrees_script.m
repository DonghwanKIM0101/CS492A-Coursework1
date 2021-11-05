if strcmp(descriptor_mode, 'K-means')
    % Test Random Forest
    leaf_assign = testTrees_fast(data_test,trees);

    for T = 1:length(trees)
        p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
    end
    % average the results from all trees
    p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
    [~,classification_result] = max(p_rf'); % Regression to Classification
    accuracy_rf = sum(classification_result==label_test')/length(classification_result); % Classification accuracy (for Caltech dataset)
    clearvars p_rf
end

if strcmp(descriptor_mode, 'RF-codebook')
    classification_result = [];
    for c = 1:10
        descriptor = cat(1, data_test{c,:});
        leaf_assign = testTrees_fast(descriptor,trees);
        number_descriptors = [];

        for T = 1:length(trees)
            p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
        end
        idx = 1;
        for i = 1:15
            p_rf_image = p_rf(idx:size(data_test{c,i},1),:,:);
            p_rf_image = squeeze(sum(p_rf_image,3))/length(trees); % Regression
            [~,predicted_labels] = max(p_rf_image'); % Regression to Classification
            classification_result(15 * (c-1) + i) = mode(predicted_labels);
        end
        clearvars p_rf
% 
%         for i = 1:15
%             descriptor = data_test{c,i};
%             leaf_assign = testTrees_fast(descriptor,trees);
% 
%             for T = 1:length(trees)
%                 p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
%             end
% 
%             % average the results from all trees
%             
%             p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
%             [~,predicted_labels] = max(p_rf'); % Regression to Classification
%             classification_result(15 * (c-1) + i) = mode(predicted_labels);
%             clearvars p_rf
%         end
    end
    accuracy_rf = sum(classification_result==label_test')/length(classification_result); % Classification accuracy (for Caltech dataset)
end