function [pr,ptau,pw,pin] = proxy_square_pts(porder)
%PROXY_SQUARE_PTS return Gpanel auss-Legendre proxy points, unit tangents, 
% weights, and function handle for determining if points are within proxy 
% surface. This function is for the proxy surface around a unit box 
% centered at the origin. The proxy points lie on the [-1.5,1.5]^2 square
%
% Input: 
%   porder - number of points on proxy surface
%   
% Output:
%   pr - (2,porder) array of proxy point locations (centered at 0 for a 
%           box of side length 1)

if nargin < 1
    porder = 64;
end

% points per panel
k = 16;

assert(mod(porder,4*k) == 0, ...
    'number of proxy points on square should be multiple of 64')

% generate panel Gauss-Legendre rule on [-1.5, 1.5]
[nds, wts] = lege.exps(k);
npanel = porder/(4*k);
panels = linspace(-1.5, 1.5, npanel + 1);
nds = reshape(panels(1:end-1) + 3/(2*npanel) * (nds + 1), 1, []);
wts = 3/(2*npanel) * repmat(wts, npanel, 1)';
one = ones(1,porder/4);

pr = [nds, one*1.5, -nds, -1.5*one;
      -1.5*one, nds, 1.5*one, -nds];

ptau = [one, one*0, -one, one*0;
        one*0, one, one*0, -one];

pw = repmat(wts, 1, 4);

pin = @(x) max(abs(x),[],1) < 1.5;

end