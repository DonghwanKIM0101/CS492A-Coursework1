load("histogram_256.mat")
data_hist_te = histogram_te;
data_hist_tr = histogram_tr;
denorm_tr = repmat(sum(data_hist_tr, 2), [1, size(data_hist_tr, 2)]);
denorm_te = repmat(sum(data_hist_te, 2), [1, size(data_hist_te, 2)]);
train = data_hist_tr ./ denorm_tr;
test = data_hist_te ./ denorm_te;

%training data set histogram
for i = 1:10
%   for j = 1:5 %only 5 images per class
    for j = 1:15
%       subplot(10,5,(i-1)*5+j) %only 5 images per class
        subplot(10,15,(i-1)*15+j)
        bar(train((i-1)*15+j, :))
        set(gca,'FontSize',5)
        ylim([0,0.04])
    end
end

%testing data set histogram
for i = 1:10
%   for j = 1:5 %only 5 images per class
    for j = 1:15
%       subplot(10,5,(i-1)*5+j) %only 5 images per class
        subplot(10,15,(i-1)*15+j)
        bar(test((i-1)*15+j, :))
        set(gca,'FontSize',5)
        ylim([0,0.04])
    end
end

