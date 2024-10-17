function [y] = cylindricalwave(a,c,k,r)
    arr = arrayfun( ...
        @(j) a(j+1) * besselh(j, 1, k*vecnorm(r - c)) .* exp(1i*j*atan2(r(2,:) - c(2), r(1,:) - c(1))), ...
        0:(length(a)-1), ...
        'UniformOutput', false ...
        );
    y = sum(cat(3, arr{:}),3);
end