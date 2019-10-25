function Phi_qff = qff_approx(data,l,qff_features)

    [d,arms] = size(data);
    qfeatures = qff_features^d;
    h_1 = hermite_rec(qff_features);
    omega_1 = roots(h_1);
    if d == 1
        omega = omega_1;
    elseif d == 2
        [X,Y] = ndgrid(omega_1,omega_1); 
        omega = [X(:) Y(:)];
    end
    h = hermite_rec(qff_features-1);
    y = h(end);
    p = 1;
    for i=length(h)-1:-1:1
        y = y + h(i) * omega.^p;
        p = p+1;
    end
    c = 2^(qff_features-1)*factorial(qff_features)/qff_features^2;
    nu_1 = c./(y.^2);
    if d == 1
        nu = nu_1;
    elseif d == 2
        nu = nu_1(:,1).*nu_1(:,2);
    end
    Phi_qff = zeros(2*qfeatures,arms);
    for i = 1 : arms
        x = data(:,i);
        Phi_qff(:,i) = QFF(x,qfeatures,l,omega,nu);
    end

end

