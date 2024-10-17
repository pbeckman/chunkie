function wts = weights(chnkr)
%WEIGHTS integration weights suitable for smooth functions defined on the 
% chunker object
%
% This is merely the standard Legendre weights scaled to the chunks
%
% Syntax: wts = weights(chnkr)
%
% Input:
%   chnkr - chunker object
%
% Output:
%   wts - smooth integration weights
%
% Examples:
%   wts = weights(chnkr)
%

% author: Travis Askham (askhamwhat@gmail.com)

  k = chnkr.k;
  nch = chnkr.nch;
  w = chnkr.wstor;
  wts = reshape(sqrt(sum((chnkr.d).^2,1)),k,nch);
  wts = wts.*bsxfun(@times,w(:),(chnkr.h(:)).');

end
