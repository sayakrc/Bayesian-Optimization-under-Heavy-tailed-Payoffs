function f = gen_rkhs_function(data,l,num_points,rkhs_index,kernel_index)
    
        [d,arms] = size(data);
        f = zeros(1,arms);
        for i = 1 : num_points
            kernel_vec = zeros(1,arms);
            x = zeros(d,1);
            for k = 1 : d
                x(k) = rand(1);
            end
            for j = 1 : arms
                y = data(:,j);
                kernel_vec(j) = kernel_function(x,y,l,kernel_index);
            end
            if rkhs_index == 1
                eta = -1 + 2.*rand(1);
            else
                eta = rand(1);
            end
            f = f + eta.*kernel_vec;
        end
    
end

