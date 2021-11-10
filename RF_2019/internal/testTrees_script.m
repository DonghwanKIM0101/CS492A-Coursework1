% Test Random Forest
leaf_assign = testTrees_fast(data_test,trees);

if strcmp(descriptor_mode, 'RF')
    for T = 1:length(trees)
        p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
    end

    % average the results from all trees
    p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
    [~,classification_result] = max(p_rf'); % Regression to Classification
    accuracy_rf = sum(classification_result==data_test(:,end)')/length(classification_result); % Classification accuracy (for Caltech dataset)
end

if strcmp(descriptor_mode, 'RF-codebook')
    idx = 1;
    cnt = 1;
    total_size = 0;
    for T = 1:length(trees)
        total_size = total_size + size(trees(T).leaf, 2);
    end
    histogram = zeros(150,total_size);
    for c = 1:10
        for i = 1:15
            hist = [];
            leaf_assign_image = leaf_assign(idx:idx+size(desc_te{c,i},2)-1,:);
            
            leaf_idx = 0;
            for T = 1:size(leaf_assign_image,2)
                for d = 1:size(leaf_assign_image,1)
                    histogram(cnt,leaf_assign_image(d,T) + leaf_idx) = histogram(cnt,leaf_assign_image(d,T) + leaf_idx) + 1;
                end
                leaf_idx = leaf_idx + size(trees(T).leaf, 2);
            end

            cnt = cnt + 1;
            idx = idx + size(desc_te{c,i},2);
        end
    end
end