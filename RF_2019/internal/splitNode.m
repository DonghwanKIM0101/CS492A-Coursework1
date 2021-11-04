function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

visualise = 0;

% Initilise child nodes
iter = param.splitNum;
if strcmp(param.weak_learner, 'axis-aligned')
    nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
    nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);
elseif strcmp(param.weak_learner, 'two-pixel')
    nodeL = struct('idx',[],'t',nan,'dimi',0,'dimj',0,'prob',[]);
    nodeR = struct('idx',[],'t',nan,'dimi',0,'dimj',0,'prob',[]);
end

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    if strcmp(param.weak_learner, 'axis-aligned')
        node.dim = 0;
    elseif strcmp(param.weak_learner, 'two-pixel')
        node.dimi = 0;
        node.dimj = 0;
    end
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];
for n = 1:iter
    
    % Split function - Modify here and try other types of split function
    
    if strcmp(param.weak_learner, 'axis-aligned')
        dim = randi(D-1); % Pick one random dimension
        d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
        d_max = single(max(data(:,dim))) - eps;
        t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
        idx_ = data(:,dim) < t;
    elseif strcmp(param.weak_learner, 'two-pixel')
        dims = randi(D-1,2,1); % Pick two random dimension
        dimi = dims(1);
        dimj = dims(2);
        range = data(:,dimi) - data(:,dimj);
        d_min = single(min(range)) + eps; % Find the data range of this dimension
        d_max = single(max(range)) - eps;
        t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
        idx_ = data(:,dimi) - data(:,dimj) < t;
    end
    
    ig = getIG(data,idx_); % Calculate information gain
    
    if visualise && strcmp(param.weak_learner, 'axis-aligned')
        visualise_splitfunc(idx_,data,dim,t,ig,n);
        pause();
    end
    
    if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
        if strcmp(param.weak_learner, 'axis-aligned')
            [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
        elseif strcmp(param.weak_learner, 'two-pixel')
            [node, ig_best, idx_best] = updateIG_two_pixel(node,ig_best,ig,t,idx_,dimi,dimj,idx_best);
        end
    end
    
end

nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise && strcmp(param.weak_learner, 'axis-aligned')
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end

function [node, ig_best, idx_best] = updateIG_two_pixel(node,ig_best,ig,t,idx,dimi,dimj,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dimi = dimi;
    node.dimj = dimj;
    idx_best = idx;
else
    idx_best = idx_best;
end
end