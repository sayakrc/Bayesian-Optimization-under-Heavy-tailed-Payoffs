function h = hermite_rec(n)
if( 0==n ), h = 1;
elseif( 1==n ), h = [2 0];
else
   h1 = zeros(1,n+1);
   h1(1:n) = 2*hermite_rec(n-1);

   h2 = zeros(1,n+1);
   h2(3:end) = 2*(n-1)*hermite_rec(n-2);

   h = h1 - h2;
end

