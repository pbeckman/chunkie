% accuracy to which we'll resolve the geometry and compute the solution
eps = 1e-6;

% which boundary value problem to solve
BC  = "Dirichlet";
% BC  = "Neumann";

% whether to solve interior or exterior problem
% region = "interior";
region = "exterior";

% whether to use Kress' kite geometry or the monstrosity with corners
geometry = "Kress";
% geometry = "corners";

%% build geometry 

cparams = [];
cparams.eps = eps;

if geometry == "Kress"
    % for smooth objects, chunkerfunc takes curve generating function with 
    % t in [0,2*pi], and parameters
    chnkr = chunkerfunc(@kite, cparams);
    chnkr = sort(chnkr);
elseif geometry == "corners"
    % for objects with corners, chunkgraph takes vertices, an adjacency matrix, 
    % curve generating functions for each edge with t in [0,1], and parameters
    verts = [
        1  1 -1 -1; 
        1 -1 -1  1
        ]; 
    adj = [
         1 -1  0  0; 
         0  1 -1  0; 
         0  0  1 -1; 
        -1  0  0  1
        ];
    curve_funcs = {@kite2}; % if not enough curve_funcs are specified, the remainder default to linear
    chnkr = chunkgraph(verts, adj, curve_funcs, cparams);
end

% plot 
figure(1)
clf
plot(chnkr, '-ko', 'markersize', 2, 'markeredgecolor', 'b')
hold on
quiver(chnkr)
set(gca, "xtick", []);
set(gca, "ytick", []);
axis tight equal

% extract weights and normals from chunkgraph object
w = weights(chnkr);
n = normals(chnkr);

%% build dense system matrix

% wavenumber
k = 10.0;

% Helmholtz double layer potential
fkernd  = @(s,t) chnk.helm2d.kern(k, s, t, 'd');
fkernsp = @(s,t) chnk.helm2d.kern(k, s, t, 'sprime');

% compute boundary to boundary double layer operator D
D  = chunkermat(chnkr, fkernd);
Sp = chunkermat(chnkr, fkernsp);

% form exterior problem system matrix K
if region == "interior"
    if BC == "Dirichlet"
        K = -0.5*eye(chnkr.npt) + D;
    elseif BC == "Neumann"
        K =  0.5*eye(chnkr.npt) + Sp;
    end
elseif region == "exterior"
    if BC == "Dirichlet"
        K =  0.5*eye(chnkr.npt) + D;
    elseif BC == "Neumann"
        K = -0.5*eye(chnkr.npt) + Sp;
    end
end

%% generate targets

ngrid = 500;
grid  = linspace(-4, 4, ngrid);
[X,Y] = meshgrid(grid, grid);
targs = [X(:).'; Y(:).'];

% a quick hack to find the exterior points
srcinfo = [];
srcinfo.sources = chnkr.r(:,:);
srcinfo.dipstr = w(:).';
srcinfo.dipvec = n(:,:); 
v    = lfmm2d(1e-8, srcinfo, 0, targs, 1).pottarg;
if region == "interior"
    inds = find(abs(abs(v) - 2*pi) < 1);
elseif region == "exterior"
    inds = find(abs(v) < 1);
end

%% compute solution

% use u_inc(x) = -ie^{-ikx*d} from Kress
d = [-1; 0];

% set up source info for fmm2d
srcinfo = [];
srcinfo.sources = chnkr.r(:,:);

if BC == "Dirichlet"
    % form Dirichlet boundary data
    u_inc = -1i*exp(-1i*k*chnkr.r(:,:).'*d);
    rhs   = -u_inc;
    
    % compute density using GMRES with dense K
    dens = gmres(K, rhs, [], eps, min(chnkr.npt, 10000));

    % give dipole strengths and directions for fmm2d double layer potential
    srcinfo.dipstr = (w(:) .* dens(:)).';
    srcinfo.dipvec = n(:,:); 
elseif BC == "Neumann"
    % form Neumann boundary data
    grad_u_inc = -d .* k*exp(-1i*k*chnkr.r(:,:).'*d).';
    rhs        = -dot(grad_u_inc, n(:,:)).';

    % compute density using GMRES with dense K
    dens = gmres(K, rhs, [], eps, min(chnkr.npt, 10000));

    % give charge strengths for fmm2d single layer potential
    srcinfo.charges = (w(:) .* dens(:)).';
end

% compute solution using appropriate layer potential using fmm2d
u = hfmm2d(eps, k, srcinfo, 0, targs(:,inds), 1).pottarg;

%% plot solution

figure(2)
clf

dt = (max(grid) - min(grid)) / ngrid;

usol = NaN * ones(size(targs,2),1);
usol(inds) = u + (region == "exterior") * -1i*exp(-1i*k*targs(:,inds).'*d).';
usol = reshape(usol, size(X));
h = pcolor(X - dt/2, Y - dt/2, real(usol));
set(h,'EdgeColor','None'); 
% cb = colorbar;
% cb.Ticks = [];
hold on
set(gca, "xtick", []);
set(gca, "ytick", []);
% if region == "exterior"
%     title(sprintf("Total field $$u_{tot} = u_{inc} + u_{sc}$$\n%s BC", BC), "Interpreter", "latex", "FontSize", 24)
% elseif region == "interior"
%     title(sprintf("Solution $$u$$\n%s BC", BC), "Interpreter", "latex", "FontSize", 24)
% end
plot(chnkr, '-k', 'linewidth', 2);
% scatter(chnkr.r(1,:), chnkr.r(2,:), 10, real(1i*exp(-1i*k*chnkr.r(:,:).'*d)), "filled");
% clim([-1,1])
axis tight equal

%%

figure(3)
clf

s = chnkr.arclengthfun;

plot(s(:), real(dens), "LineWidth", 3, "Color", "red");
hold on
plot(s(:), imag(dens), "LineWidth", 3, "Color", "blue");
xlim([0, max(s(:))]);
xlabel("$$s$$", "Interpreter", "latex", "FontSize", 20)
ylabel("$$\sigma$$", "Interpreter", "latex", "FontSize", 20)

%% curve functions

% kite shape from 'On the numerical solution of a hypersingular integral 
% equation in scattering theory' by Kress 1994
function [r, d, d2] = kite(t)
    t  = reshape(t, [], length(t));
    r  = [cos(t) + 0.65*cos(2*t) - 0.65; 1.5*sin(t)];
    d  = [-sin(t) - 1.3*sin(2*t); 1.5*cos(t)];
    d2 = [-cos(t) - 2.6*cos(2*t); -1.5*sin(t)];
end

% rescaled half kites so that t=[0,1] -> [0,pi] and t=[0,1] -> [pi,2*pi] respectively
function [r, d, d2] = kite1(t)
    [r, d, d2] = kite(pi*t); d = pi*d; d2 = pi^2*d2;
end
function [r, d, d2] = kite2(t)
    [r, d, d2] = kite(pi*t + pi); d = pi*d; d2 = pi^2*d2;
end

