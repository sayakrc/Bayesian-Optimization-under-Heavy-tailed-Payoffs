function phi = QFF(x,m,l,omega,nu)
    
    x = x*(sqrt(2)/l);
    phi = zeros(2*m,1);
    for i = 1 : m
        phi(i) = sqrt(nu(i))*cos(omega(i,:)*x);
        phi(m+i) = sqrt(nu(i))*sin(omega(i,:)*x);
    end
end