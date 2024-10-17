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

% discretize domain

tol = 1e-10;

cparams = [];
cparams.eps = tol;

pref = []; 
pref.k = 16; 
pref.dim = 2;
pref.nchmax = 100000;

start = tic;

% starfish
cparams.nover = 0;
narms = 5;
amp = 0.25;
chnkr = chunkerfunc(@(t) starfish(t,narms,amp),cparams,pref);

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

fkern = @(s,t) chnk.lap2d.kern(s,t,'D')+1;
opdims(1) = 1; opdims(2) = 1;
% opts = [];
opts = struct('eps', tol);
start = tic; sysmat = chunkermat(chnkr,fkern,opts);
t1 = toc(start);

fprintf('%5.2e s : time to assemble matrix\n',t1)

K = 0.5*eye(chnkr.k*chnkr.nch) + sysmat;

% find targets

rmin = min(chnkr); rmax = max(chnkr);
xl = rmax(1)-rmin(1);
yl = rmax(2)-rmin(2);
nplot = 200;
xtarg = linspace(rmin(1)-xl,rmax(1)+xl,nplot); 
ytarg = linspace(rmin(2)-yl,rmax(2)+yl,nplot);
[xxtarg,yytarg] = meshgrid(xtarg,ytarg);
targets = zeros(2,length(xxtarg(:)));
targets(1,:) = xxtarg(:); targets(2,:) = yytarg(:);

start = tic; in = chunkerinterior(chnkr,targets); t1 = toc(start);
out = ~in;

fprintf('%5.2e s : time to find points in domain\n',t1)

extargets = targets(:,out);
x0 = extargets(:,10150);

% rhs = zeros(chnkr.nch*chnkr.k,1);
% srcinfo = struct("r", x0);
% for i=1:chnkr.nch
%     for j=1:chnkr.k
%         targinfo    = []; 
%         targinfo.r  = chnkr.r(:,j,i); 
%         targinfo.d  = chnkr.d(:,j,i); 
%         targinfo.d2 = chnkr.d2(:,j,i);
%         rhs((i-1)*chnkr.k + j) = fkern(targinfo, srcinfo);
%     end
% end
A = chunkerkernevalmat(chnkr,fkern,targets(:,out),opts);
rhs = A(10150,:)';
start = tic; mu  = gmres(K',rhs,[],1e-13,100); t1 = toc(start);
fprintf('%5.2e s : time for dense gmres on K^T \n',t1)
start = tic; sig = gmres(K, mu, [],1e-13,100); t1 = toc(start);
fprintf('%5.2e s : time for dense gmres on K \n',t1)

%%

% compute layer potential based on oversample boundary

start = tic;
opts = struct("eps", 1e-6);
u = chunkerkerneval(chnkr,fkern,sig,targets(:,out),opts); t1 = toc(start);
fprintf('%5.2e s : time for kernel eval (for plotting)\n',t1)

% compute variance of solution with iid Gaussian noise

%% Plot everything

uplot = u;

maxu = max(abs(uplot(:)));

clf
t = tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = uplot;
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'LineWidth',2)
scatter(x0(1),x0(2),'filled','SizeData',50,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','white')
axis equal
axis tight
colormap(redblue)
caxis([-maxu,maxu])
if j == 2
    title('$Cov(u(x_0),u(x))$','Interpreter','latex','FontSize',20)
end

% exportgraphics(t,'/Users/beckman/Documents/Research/IE-UQ/output/accurate-3.pdf','backgroundcolor','none')

