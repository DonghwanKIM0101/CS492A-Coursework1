% Test Random Forest
leaf_assign = testTrees_fast(data_test,trees);

for T = 1:length(trees)
    p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
end

if strcmp(descriptor_mode, 'K-means')
    % average the results from all trees
    p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
    [~,classification_result] = max(p_rf'); % Regression to Classification
    accuracy_rf = sum(classification_result==label_test')/length(classification_result); % Classification accuracy (for Caltech dataset)
    clearvars p_rf
end

if strcmp(descriptor_mode, 'RF-codebook')
    classification_result = [];

    idx = 1;
    for c = 1:10
        for i = 1:15
            p_rf_image = p_rf(idx:idx+size(desc_te{c,i},2)-1,:,:);
            p_rf_image = squeeze(sum(p_rf_image,3))/length(trees); % Regression
            [~,predicted_labels] = max(p_rf_image'); % Regression to Classification
            classification_result(15 * (c-1) + i) = mode(predicted_labels);
            idx = idx + size(desc_te{c,i},2);
        end
    end
    
    accuracy_rf = sum(classification_result==label_test')/length(classification_result); % Classification accuracy (for Caltech dataset)
    clearvars p_rf
end