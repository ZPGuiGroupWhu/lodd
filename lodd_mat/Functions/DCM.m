function [int_id, bou_id] = DCM(X, k_num, ratio)
[n,d] = size(X);
if(d>5)
    X = pca(X,2);
    d = 5;
end
[get_knn, ~] = knnsearch(X,X,'k',k_num+1);
get_knn(:,1) = [];
DCM = zeros(n,1);
if (d==2)
    angle = zeros(n,k_num);
    for i=1:n
        for j=1:k_num
            delta_x = X(get_knn(i,j),1)-X(i,1);
            delta_y = X(get_knn(i,j),2)-X(i,2);
            if(delta_x==0)
                if(delta_y==0)
                    angle(i,j)=0;
                elseif(delta_y>0)
                    angle(i,j)=pi/2;
                else
                    angle(i,j)=3*pi/2; 
                end
            elseif(delta_x>0)
                if(atan(delta_y/delta_x)>=0)
                    angle(i,j)=atan(delta_y/delta_x);
                else
                    angle(i,j)=2*pi+atan(delta_y/delta_x);
                end
            else
                angle(i,j)=pi+atan(delta_y/delta_x);
            end
        end
    end                             
    for i=1:n
        angle_order = sort(angle(i,:));
        for j=1:k_num-1
            point_angle = angle_order(j+1)-angle_order(j);
            DCM(i) = DCM(i) + (point_angle-2*pi/k_num).^2;
        end
        point_angle = angle_order(1)-angle_order(k_num)+2*pi;
        DCM(i) = DCM(i) + (point_angle-2*pi/k_num).^2;
        DCM(i) = DCM(i)/k_num;
    end   
    DCM = DCM/((k_num-1)*4*pi^2/k_num^2);    
else
    for i=1:n
        try
            dif_x = X(get_knn(i,:),:) - X(i,:);
            map_x = inv(diag(sqrt(diag(dif_x*dif_x'))))*dif_x;
            convex = convhulln(map_x);
            simplex_num = length(convex(:,1));
            simplex_vol = zeros(simplex_num,1);
            for j=1:simplex_num
                simplex_coord = map_x(convex(j,:),:);
                simplex_vol(j) = sqrt(max(0,det(simplex_coord*simplex_coord')))/gamma(d-1);
            end  
            DCM(i) = var(simplex_vol);
        catch exception 
            DCM(i) = 0;
        end
    end
end
[~, bou_id] = maxk(DCM, round(n*ratio));
int_id = setdiff(1:n, bou_id);
end