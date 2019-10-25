function k = kernel_function(x,y,l,index)

   r = norm((x-y),2);
  
   if index == 1
     k = exp(-(r^2)/(2*l^2)); % squared-exponential-kernel
   else
     k = (1 + (sqrt(5)*r/(l)) + (5*(r^2)/(3*((l)^2))))*exp(-(sqrt(5)*r)/l); % matern kernel (nu=2.5)
   end
   
   end

