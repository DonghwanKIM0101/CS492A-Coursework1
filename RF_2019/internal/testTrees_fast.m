function label = testTrees_fast(data,tree)
% Faster version - pass all data at same time
cnt = 1;
for T = 1:length(tree)
    idx{1} = 1:size(data,1);
    for n = 1:length(tree(T).node);
        if (strcmp(tree(T).weak_learner, 'axis-aligned') && ~tree(T).node(n).dim) || (strcmp(tree(T).weak_learner, 'two-pixel') && (~tree(T).node(n).dimi || ~tree(T).node(n).dimj))
            leaf_idx = tree(T).node(n).leaf_idx;
            if ~isempty(tree(T).leaf(leaf_idx))
                if strcmp(tree(T).descriptor_mode, 'RF')
                    label(idx{n}',T) = tree(T).leaf(leaf_idx).label;
                elseif strcmp(tree(T).descriptor_mode, 'RF-codebook')
                    label(idx{n}',T) = leaf_idx;
                end
            end
            continue;
        end
        if (strcmp(tree(T).weak_learner, 'axis-aligned'))
            idx_left = data(idx{n},tree(T).node(n).dim) < tree(T).node(n).t;
        elseif strcmp(tree(T).weak_learner, 'two-pixel')
            idx_left = data(idx{n},tree(T).node(n).dimi) - data(idx{n},tree(T).node(n).dimj) < tree(T).node(n).t;
        end
        idx{n*2} = idx{n}(idx_left');
        idx{n*2+1} = idx{n}(~idx_left');
    end
end

end

