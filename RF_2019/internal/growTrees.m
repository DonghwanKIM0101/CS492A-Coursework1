function tree = growtrees(data,param)

%        Base               Each node stores:
%         1                   trees.idx       - data (index only) which split into this node
%        / \                  trees.t         - threshold of split function
%       2   3                 trees.dim       - feature dimension of split function
%      / \ / \                trees.prob      - class distribution of this node
%     4  5 6  7               trees.leaf_idx  - leaf node index (empty if it is not a leaf node) 
% 
% trees.dim for param.weak_learner, 'axis-aligned'
% trees.dimi, trees.dimj for param_weak_learner, 'two-pixel'
% 
% tree stores:
% tree.weak_learner = param.weak_learner

disp('Training Random Forest...');

[N,D] = size(data);
frac = 1 - 1/exp(1); % Bootstrap sampling fraction: 1 - 1/e (63.2%)

cnt_total = 1;
[labels,~] = unique(data(:,end));
for T = 1:param.num
    fprintf('Training tree: %i\n', T)
    % Bootstraping aggregating
    idx = randsample(N,ceil(N*frac),1); % A new training set for each tree is generated by random sampling from dataset WITH replacement.
    prior = histc(data(idx,end),labels)/length(idx);
    
    tree(T).weak_learner = param.weak_learner;
    tree(T).descriptor_mode = param.descriptor_mode;
    % Initialise base node
    if strcmp(param.weak_learner, 'axis-aligned')
        tree(T).node(1) = struct('idx',idx,'t',nan,'dim',-1,'prob',[]);
    elseif strcmp(param.weak_learner, 'two-pixel')
        tree(T).node(1) = struct('idx',idx,'t',nan,'dimi',-1,'dimj',-1,'prob',[]);
    end
    
    % Split Nodes
    for n = 1:2^(param.depth-1)-1
        [tree(T).node(n),tree(T).node(n*2),tree(T).node(n*2+1)] = splitNode(data,tree(T).node(n),param);
    end
    
    % Leaf Nodes
    cnt = 1;
    for n = 1:2^param.depth-1
        if ~isempty(tree(T).node(n).idx)
            % Percentage of observations of each class label
            tree(T).node(n).prob = histc(data(tree(T).node(n).idx,end),labels)/length(tree(T).node(n).idx);
            
            if (strcmp(param.weak_learner, 'axis-aligned') && ~tree(T).node(n).dim) || (strcmp(param.weak_learner, 'two-pixel') && (~tree(T).node(n).dimi || ~tree(T).node(n).dimj)) % if this is a leaf node
                tree(T).node(n).leaf_idx = cnt;
                tree(T).leaf(cnt).label = cnt_total;
                prob = reshape(histc(data(tree(T).node(n).idx,end),labels),[],1);

                tree(T).leaf(cnt).prob = prob; %.*prior; % Multiply by the prior probability of bootstrapped sub-training-set
                tree(T).leaf(cnt).prob = tree(T).leaf(cnt).prob./sum(tree(T).leaf(cnt).prob); % Normalisation
                
                if strcmp(param.split,'Var')
                    tree(1).cc(cnt_total,:) = mean(data(tree(T).node(n).idx,1:end-1),1); % For RF clustering, unfinished
                else
                    tree(1).prob(cnt_total,:) = tree(T).node(n).prob';
                end
                
                cnt = cnt+1;
                cnt_total = cnt_total + 1;
            end
        end
    end
end
end