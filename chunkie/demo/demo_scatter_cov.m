%DEMO_SCATTER
%
% Define an exterior scattering problem on a starfish-shaped domain and 
% solve
%

%% Set up geometry

clearvars; close all;
iseed = 8675309;
rng(iseed);
addpaths_loc();

% planewave vec

kvec = 5*[1;-1.5];

%

zk = norm(kvec);

% discretize domain

cparams = [];
cparams.eps = 1.0e-7;
cparams.nover = 0;
cparams.maxchunklen = 4.0/zk; % setting a chunk length helps when the
                              % frequency is known
pref = []; 
pref.k = 16;
narms = 5;
amp = 0.25;
start = tic; chnkr = chunkerfunc(@(t) starfish(t,narms,amp),cparams,pref); 
t1 = toc(start);

fprintf('%5.2e s : time to build geo\n',t1)

[~,~,info] = sortinfo(chnkr);
assert(info.ier == 0);

% plot geometry and data

% figure(1)
% clf
% plot(chnkr,'-x')
% hold on
% quiver(chnkr)
% axis equal

% solve and visualize the solution

%% Build CFIE

fkern = @(s,t) chnk.helm2d.kern(zk,s,t,'c',1);
opdims(1) = 1; opdims(2) = 1;
opts = [];
start = tic; sysmat = chunkermat(chnkr,fkern,opts);
t1 = toc(start);

fprintf('%5.2e s : time to assemble matrix\n',t1)

sys = 0.5*eye(chnkr.k*chnkr.nch) + sysmat;

rhs = -planewave(kvec(:),chnkr.r(:,:)); rhs = rhs(:);
start = tic; sol = gmres(sys,rhs,[],1e-13,100); t1 = toc(start);

fprintf('%5.2e s : time for dense gmres\n',t1)

% evaluate at targets and plot

rmin = min(chnkr); rmax = max(chnkr);
xl = rmax(1)-rmin(1);
yl = rmax(2)-rmin(2);
nplot = 100;
xtarg = linspace(rmin(1)-xl,rmax(1)+xl,nplot); 
ytarg = linspace(rmin(2)-yl,rmax(2)+yl,nplot);
[xxtarg,yytarg] = meshgrid(xtarg,ytarg);
targets = zeros(2,length(xxtarg(:)));
targets(1,:) = xxtarg(:); targets(2,:) = yytarg(:);

%

start = tic; in = chunkerinterior(chnkr,targets); t1 = toc(start);
out = ~in;

fprintf('%5.2e s : time to find points in domain\n',t1)

% compute layer potential based on oversample boundary

start = tic;
opts = struct("flam", 0, "forcesmooth", 1);
[uscat, kernmat] = chunkerkerneval(chnkr,fkern,sol,targets(:,out),opts); t1 = toc(start);
fprintf('%5.2e s : time for kernel eval (for plotting)\n',t1)

uin = planewave(kvec,targets(:,out));
utot = uscat(:)+uin(:);

% compute variance of solution with iid Gaussian noise

%%

S = eye(chnkr.npt);

% k = @(xi, xj) exp(-norm(xi-xj)^2);
% for i = 1:chnkr.npt
%     for j = 1:chnkr.npt
%         S(i,j) = k(...
%             chnkr.r(:, rem(i-1,chnkr.k)+1, fix((i-1)/chnkr.k)+1), ...
%             chnkr.r(:, rem(j-1,chnkr.k)+1, fix((j-1)/chnkr.k)+1)  ...
%         );
%         if i == j
%             S(i,j) = S(i,j) + 1e-12;
%         end
%     end
% end
% 
% clf
% samples = chol(S)'*randn(chnkr.npt, 1);
% hold on
% for ch = 1:chnkr.nch
%     plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), zeros(chnkr.k), 'color', 'blue');
%     plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), samples((ch-1)*chnkr.k+1:ch*chnkr.k), 'color', 'red');
% end
% % zlim([-0.2, 0.2]);
% hold off

%%

nt = length(uscat);
i0s = [
    nplot*round(nplot/4) + round(nplot/4),
    nplot*round(nplot/3) + round(2*nplot/3),
    nplot*round(5*nplot/6) + round(nplot/6)
    ];
uvar  = zeros(nt,1);
u0cov = zeros(3,nt);
v0  = zeros(3,length(sys));
for j = 1:3
    v0(j,:)  = (sys.')\(kernmat(i0s(j),:).');
end
for i = 1:nt
    v = (sys.')\(kernmat(i,:).');
    uvar(i) = 0.5*real(dot(v, S*conj(v)) + dot(v, S*v));
%     uvar(i) = dot(v, v);
    for j=1:3
        u0cov(j,i) = 0.5*real(dot(v0(j,:), S*conj(v)) + dot(v0(j,:), S*v));
%         u0cov(j,i) = dot(v0(j,:), v);
    end
end
% save('udist.mat', 'uvar', 'u0cov');
% load('udist.mat');

maxu = max(abs(uin(:)));

%% Plot everything

% figure(2)
clf
subplot(2,3,1)
zztarg = nan(size(xxtarg));
zztarg(out) = uin;
h=pcolor(xxtarg,yytarg,real(zztarg));
set(h,'EdgeColor','none')
hold on
plot(chnkr,'LineWidth',2)
axis equal
axis tight
colormap(redblue)
caxis([-maxu,maxu])
title('$u_{in}$','Interpreter','latex','FontSize',24)

subplot(2,3,2)
zztarg = nan(size(xxtarg));
zztarg(out) = uscat;
h=pcolor(xxtarg,yytarg,real(zztarg));
set(h,'EdgeColor','none')
hold on
plot(chnkr,'LineWidth',2)
axis equal
axis tight
colormap(redblue)
caxis([-maxu,maxu])
title('$E[u_{scat}]$','Interpreter','latex','FontSize',24)

vec = [100; 90; 50; 10; 0];
hex = ['#661fa3'; '#661fa3'; '#ffffff'; '#219e1c'; '#219e1c'];
raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
N = 128;
greenpurple = interp1(vec,raw,linspace(100,0,N),'pchip');
uvarmax  = 0.07*max(uvar(:)); 
u0covmax = max(max(abs([min(u0cov(:)), max(u0cov(:))])));

ax = subplot(2,3,3);
zztarg = nan(size(xxtarg));
zztarg(out) = uvar;
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'LineWidth',2)
axis equal
axis tight
colormap(ax,redblue)
caxis([-uvarmax,uvarmax])
title('$Var(u_{scat})$','Interpreter','latex','FontSize',20)

tout = targets(:,out);

for j = 1:3
    ax = subplot(2,3,3+j);
    zztarg = nan(size(xxtarg));
    zztarg(out) = u0cov(j,:);
    h=pcolor(xxtarg,yytarg,zztarg);
    set(h,'EdgeColor','none')
    hold on
    plot(chnkr,'LineWidth',2)
    scatter(tout(1,i0s(j)),tout(2,i0s(j)),'filled','SizeData',50,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','white')
    axis equal
    axis tight
    colormap(ax,redblue)
    caxis([-u0covmax,u0covmax])
    if j == 2
        title('$Cov(u_{scat}(x_0),u_{scat}(x))$','Interpreter','latex','FontSize',20)
    end
end
