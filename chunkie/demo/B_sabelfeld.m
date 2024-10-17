function cov = B_sabelfeld(k_per, xi, xj) 
    rho = norm(xi) * norm(xj);
    psi = atan(xi(2)/xi(1)) - atan(xj(2)/xj(1));
    integrand = @(dum) arrayfun(k_per, dum) ./ (1 - 4*rho/(1-rho)^2 * arrayfun(@sin, (psi - dum) ./ 2) .^ 2); 

    dums = linspace(0, 2*pi, 1000);
    clf
    plot(dums, integrand(dums));

    cov = 1/(2*pi) * (1-rho^2)/(1-rho)^2  * integral(integrand, 0, 2*pi);
end