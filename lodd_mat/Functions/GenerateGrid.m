n = 35;
X = rand(n,2);
gnum = ceil(sqrt(n));
marker = ceil(n/gnum):ceil(n/gnum):n;
if(mod(n,gnum)~=0)
    marker = [0,marker,n];
else
    marker = [0,marker];
end
[sort_x, x_id] = sort(X(:,1));

figure(1)
for i = 2:length(marker)-1
    plot([0.5*(sort_x(marker(i))+sort_x(marker(i)+1)), 0.5*(sort_x(marker(i))+sort_x(marker(i)+1))], [0, 1], 'k');
    hold on;
end

xx = {};
for i = 1:length(marker)-1
    subset_x = X(x_id(marker(i)+1:marker(i+1)),:);
    [~,y_id] = sort(subset_x(:,2));
    for j = 1:length(y_id)-1
        plot(subset_x([y_id(j),y_id(j+1)],1),subset_x([y_id(j),y_id(j+1)],2),'b-');
        hold on;
    end
    xx{i} = subset_x(y_id,:);
end

for i = 1:marker(2)
    temp_x = [];
    for j = 1:length(marker)-1
        if(size(xx{j},1)>=i)
            temp_x = [temp_x;xx{j}(i,:)];
        end
    end
    for j = 1:length(temp_x)-1
        plot(temp_x([j,j+1],1),temp_x([j,j+1],2),'b-');
        hold on;
    end
end

plot(X(:,1),X(:,2),'bo');
hold on;
xlim([0, 1])
ylim([0, 1])

figure(2)

mod_y = 0.1+(0.8./(marker(2)-1))*(0:marker(2)-1);
mod_x_set = [];
for i = 1:length(marker)-1
    npts = marker(i+1)-marker(i);
    mod_x = mean(xx{i},1);
    mod_x_set = [mod_x_set;mod_x(1)];
    plot(mod_x(1)*ones(npts,1),mod_y(1:npts),'bo');
    hold on;
    plot([mod_x(1),mod_x(1)],[mod_y(1),mod_y(npts)],'b-');
    hold on;
end
for i = 1:marker(2)
    if(i<=npts)
        plot([mod_x_set(1),mod_x_set(end)],[mod_y(i),mod_y(i)],'b-');
        hold on;
    else
        plot([mod_x_set(1),mod_x_set(end-1)],[mod_y(i),mod_y(i)],'b-');
        hold on;
    end
end
xlim([0, 1])
ylim([0, 1])