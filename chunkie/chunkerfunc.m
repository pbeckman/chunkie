function chnkr = chunkerfunc(fcurve,cparams,pref)
%CHUNKERFUNC create a chunker object corresponding to a parameterized curve
%
% Syntax: chnkr = chunkerfunc(fcurve,cparams,pref)
%
% Input: 
%   fcurve - function handle of the form
%               [r,d,d2] = fcurve(t)
%            where r, d, d2 are size [dim,size(t)] arrays describing
%            position, first derivative, and second derivative of a curve
%            in dim dimensions parameterized by t.
%
% Optional input:
%	cparams - curve parameters structure (defaults)
%       cparams.ta = left end of t interval (0)
%       cparams.tb = right end of t interval (2*pi)
%       cparams.ifclosed = flag determining if the curve
%           is to be interpreted as a closed curve (true)
%       cparams.chsmall = max size of end intervals if
%           ifclosed == 0 (Inf)
%       cparams.nover = oversample resolved curve nover
%           times (0)
%       cparams.eps = resolve coordinates, arclength,
%          and first and second derivs of coordinates
%          to this tolerance (1.0e-6)
%       cparams.lvlr = string, determines type of level
%          restriction to be enforced
%               lvlr = 'a' -> no chunk should have double the arc length 
%                               of its neighbor 
%               lvlr = 't' -> no chunk should have double the length in 
%                               parameter space of its neighbor 
%               lvlr = 'n' -> no enforcement of level restriction
%       cparams.lvlrfac = factor in level restriction, i.e. check if 
%               neighboring chunks differ in size by this factor (2.0)
%       cparams.maxchunklen - maximum length of any chunk (Inf)
%   pref - chunkerpref object or structure (defaults)
%       pref.nchmax - maximum number of chunks (10000)
%       pref.k - number of Legendre nodes on chunks (16)
%
% Examples:
%   chnkr = chunkerfunc(@(t) starfish(t)); % chunk up starfish w/ standard
%                                        % options
%   pref = []; pref.k = 30; 
%   cparams = []; cparams.eps = 1e-3;
%   chnkr = chunkerfunc(@(t) starfish(t),cparams,pref); % change up options
%   
% see also CHUNKERPOLY, CHUNKERPREF, CHUNKER

% author: Travis Askham (askhamwhat@gmail.com)
%

if nargin < 2
    cparams = [];
end
if nargin < 3
    pref = chunkerpref();
else
    pref = chunkerpref(pref);
end


ta = 0.0; tb = 2*pi; ifclosed=true;
chsmall = Inf; nover = 0;
eps = 1.0e-6;
lvlr = 'a'; maxchunklen = Inf; lvlrfac = 2.0;

if isfield(cparams,'ta')
    ta = cparams.ta;
end	 
if isfield(cparams,'tb')
    tb = cparams.tb;
end	 
if isfield(cparams,'ifclosed')
    ifclosed = cparams.ifclosed;
end	 
if isfield(cparams,'chsmall')
    chsmall = cparams.chsmall;
end	 
if isfield(cparams,'nover')
    nover = cparams.nover;
end	 
if isfield(cparams,'eps')
    eps = cparams.eps;
end	 
if isfield(cparams,'lvlr')
    lvlr = cparams.lvlr;
end
if isfield(cparams,'lvlrfac')
    lvlrfac = cparams.lvlrfac;
end
if isfield(cparams,'maxchunklen')
    maxchunklen = cparams.maxchunklen;
end

k = pref.k;
nchmax = pref.nchmax; 
 
dim = checkcurveparam(fcurve,ta);
pref.dim = dim;
nout = 3;
out = cell(nout,1);


ifprocess = zeros(nchmax,1);

%       construct legendre nodes and weights, k and 2k of them, as well
%       as the interpolation/coefficients matrices

k2 = 2*k;
[xs,ws,us,vs] = lege.exps(k);
[xs2,ws2,u2] = lege.exps(k2);   

xs2p = ((1:k2)-1)/(k2-1)*2-1;
[polvals,~] = lege.pols(xs2p,k-1);
interp_xs = reshape(polvals,[k,k2]).'*us; 



%       . . . start chunking

ab = zeros(2,nchmax);
adjs = zeros(2,nchmax);
ab(1,1)=ta;
ab(2,1)=tb;
nch=1;
if ifclosed
    adjs(1,1)=1;
    adjs(2,1)=1;
else
    adjs(1,1)=-1;
    adjs(2,1)=-1;
end
nchnew=nch;

maxiter_res=10000;

rad_curr = 0;
for ijk = 1:maxiter_res

%       loop through all existing chunks, if resolved store, if not split
    xmin =  Inf;
    xmax = -Inf;
    ymin =  Inf;
    ymax = -Inf;
    
    ifdone=1;
    for ich=1:nchnew

        if (ifprocess(ich) ~= 1)
            ifprocess(ich)=1;

            a=ab(1,ich);
            b=ab(2,ich);
            rlself = chunklength(fcurve,a,b,xs,ws);
           
            ts = a + (b-a)*(xs2+1)/2.0;
            [r,d,d2] = fcurve(ts);
            
            zd = d(1,:)+1i*d(2,:);
            vd = abs(zd);
            zdd= d2(1,:)+1i*d2(2,:);
            dkappa = imag(zdd.*conj(zd))./abs(zd).^2;

            cfs = u2*vd.';
            errs0 = sum(abs(cfs(1:k)).^2,1);
            errs = sum(abs(cfs(k+1:k2)).^2,1);
            err1 = sqrt(errs/errs0/k);
            
            resol_speed_test = err1>eps;
            
            xmax = max(xmax,max(r(1,:)));
            ymax = max(ymax,max(r(2,:)));
            xmin = min(xmin,min(r(1,:)));
            ymin = min(ymin,min(r(2,:)));
            
            cfsx = u2*r(1,:).';
            cfsy = u2*r(2,:).';
            errx = sum(abs(cfsx(k+1:k2)).^2/k,1);
            erry = sum(abs(cfsy(k+1:k2)).^2/k,1);
            errx = sqrt(errx);
            erry = sqrt(erry);
            
            resol_curve_test = true;
            
            if (ijk >1)
                if (errx/rad_curr<eps && erry/rad_curr<eps)
                    resol_curve_test = false;
                end
            end    


           total_curve = (b-a)/2*sum(abs(dkappa).*ws2.');
           total_curve_test = total_curve >= (2*pi)/3;

           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            

     %       . . . mark as processed and resolved if less than eps

            if (resol_speed_test || resol_curve_test  || ... 
                   total_curve_test || rlself > maxchunklen || ...
                    and(or(adjs(1,ich) <= 0, adjs(2,ich) <= 0), ...
                rlself > chsmall))
              %       . . . if here, not resolved
              %       divide - first update the adjacency list
                if (nch +1 > nchmax)
                    error('too many chunks')
                end

                ifprocess(ich)=0;
                ifdone=0;

                if ((nch == 1) && ifclosed)
                    adjs(1,nch)=2;
                    adjs(2,nch)=2;
                    adjs(1,nch+1)=1;
                    adjs(2,nch+1)=1;
                end

                if ((nch == 1) && (~ifclosed))
                    adjs(1,nch)=-1;
                    adjs(2,nch)=2;
                    adjs(1,nch+1)=1;
                    adjs(2,nch+1)=-1;
                end

                if (nch > 1)
                    iold2=adjs(2,ich);
                    adjs(2,ich)=nch+1;
                    if (iold2 > 0)
                        adjs(1,iold2)=nch+1;
                    end	
                    adjs(1,nch+1)=ich;
                    adjs(2,nch+1)=iold2;

                end
                    %       now update the endpoints in ab

                ab(1,ich)=a;
                ab(2,ich)=(a+b)/2;

                nch=nch+1;

                ab(1,nch)=(a+b)/2;
                ab(2,nch)=b;
            end
        end
    end
    if ((ifdone == 1) && (nchnew == nch))
        break;
    end
    nchnew=nch;
    
    rad_curr = max(xmax-xmin,ymax-ymin);
end


%       the curve should be resolved to precision eps now on
%       each interval ab(,i)
%       check the size of adjacent neighboring chunks - if off by a
%       factor of more than 2, split them as well. iterate until done.
   
if or(strcmpi(lvlr,'a'),strcmpi(lvlr,'t'))
    maxiter_adj=1000;
    for ijk = 1:maxiter_adj

        nchold=nch;
        ifdone=1;
        for i = 1:nchold
            i1=adjs(1,i);
            i2=adjs(2,i);

    %       calculate chunk lengths

            a=ab(1,i);
            b=ab(2,i);
            
            if strcmpi(lvlr,'a')
                rlself = chunklength(fcurve,a,b,xs,ws);

                rl1=rlself;
                rl2=rlself;

                if (i1 > 0)
                    a1=ab(1,i1);
                    b1=ab(2,i1);
                    rl1 = chunklength(fcurve,a1,b1,xs,ws);
                end
                if (i2 > 0)
                    a2=ab(1,i2);
                    b2=ab(2,i2);
                    rl2 = chunklength(fcurve,a2,b2,xs,ws);
                end
            else
                
                rlself = b-a;
                rl1 = rlself;
                rl2 = rlself;
                if (i1 > 0)
                    rl1 = ab(2,i1)-ab(1,i1);
                end
                if (i2 > 0)
                    rl2 = ab(2,i2)-ab(1,i2);
                end
            end

    %       only check if self is larger than either of adjacent blocks,
    %       iterating a couple times will catch everything

            if (rlself > lvlrfac*rl1 || rlself > lvlrfac*rl2)

    %       split chunk i now, and recalculate nodes, ders, etc

                if (nch + 1 > nchmax)
                    error('too many chunks')
                end


                ifdone=0;
                a=ab(1,i);
                b=ab(2,i);
                ab2=(a+b)/2;

                i1=adjs(1,i);
                i2=adjs(2,i);
    %        
                adjs(1,i) = i1;
                adjs(2,i) = nch+1;

    %       . . . first update nch+1

                adjs(1,nch+1) = i;
                adjs(2,nch+1) = i2;

     %       . . . if there's an i2, update it

                if (i2 > 0)
                    adjs(1,i2) = nch+1;
                end

                nch=nch+1;
                ab(1,i)=a;
                ab(2,i)=ab2;

                ab(1,nch)=ab2;
                ab(2,nch)=b;
            end
        end

        if (ifdone == 1)
            break;
        end

    end
end

%       go ahead and oversample by nover, updating
%       the adjacency information adjs along the way


if (nover > 0) 
    for ijk = 1:nover

        nchold=nch;
        for i = 1:nchold
            a=ab(1,i);
            b=ab(2,i);
		   %       find ab2 using newton such that 
		   %       len(a,ab2)=len(ab2,b)=half the chunk length
            rl = chunklength(fcurve,a,b,xs,ws);
            rlhalf=rl/2;
            thresh=1.0d-8;
            ifnewt=0;
            ab0=(a+b)/2;
            for iter = 1:1000

                [rl1] = chunklength(fcurve,a,ab0,xs,ws);
                
                [out{:}] = fcurve(ab0);
                dsdt = sqrt(sum((abs(out{2})).^2));
                ab1=ab0-(rl1-rlhalf)/dsdt;

                err=rl1-rlhalf;
                if (abs(err) < thresh)
                    ifnewt=ifnewt+1;
                end

                if (ifnewt == 3)
                    break;
                end
                ab0=ab1;
            end
	 
            if (ifnewt < 3) 
                error('newton failed in chunkerfunc');
            end
            ab2=ab1;

            i1=adjs(1,i);
            i2=adjs(2,i);
            adjs(2,i)=nch+1;
            if (i2 > 0)
                adjs(1,i2)=nch+1;
            end

            if (nch + 1 > nchmax)
                error('too many chunks')
            end

            adjs(1,nch+1)=i;
            adjs(2,nch+1)=i2;
	 
            ab(1,i)=a;
            ab(2,i)=ab2;
	 
            nch=nch+1;

            ab(1,nch)=ab2;
            ab(2,nch)=b;
        end
    end
end

%       up to here, everything has been done in parameter space, [ta,tb]
%       . . . finally evaluate the k nodes on each chunk, along with 
%       derivatives and chunk lengths

chnkr = chunker(pref); % empty chunker
chnkr = chnkr.addchunk(nch);


for i = 1:nch
    a=ab(1,i);
    b=ab(2,i);
    
    ts = a + (b-a)*(xs+1)/2;
    [out{:}] = fcurve(ts);
    chnkr.r(:,:,i) = reshape(out{1},dim,k);
    chnkr.d(:,:,i) = reshape(out{2},dim,k);
    chnkr.d2(:,:,i) = reshape(out{3},dim,k);
    chnkr.h(i) = (b-a)/2;
end

chnkr.adj = adjs(:,1:nch);

% update normals
chnkr.n = normals(chnkr);

end


function [len] = chunklength(fcurve,a,b,xs,ws)
    
    nout = 3;
    out = cell(nout,1);
    ts = a+(b-a)*(xs+1)/2;
    [out{:}] = fcurve(ts);
    dsdt = sqrt(sum(abs(out{2}).^2,1));
    len = dot(dsdt,ws)*(b-a)/2;
 end

