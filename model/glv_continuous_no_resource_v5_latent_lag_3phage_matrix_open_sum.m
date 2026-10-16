function dydt = glv_continuous_no_resource_v5_latent_lag_3phage_matrix_open_sum(t, y, parameters)
%
% y ：
%   [Abi(1..n); nonAbi(1..n); Resistant(1..n);    % 3n
%    L_Abi_k(1..n) for k=1..K;                     % nK
%    L_nonAbi_k(1..n) for k=1..K;                  %  nK
%    P(1..K);                                      % K
%    Q]                                            
%
% length：3n + 2nK + K + 1

% ===== 0. initialisation =====
persistent firstCall
if isempty(firstCall); firstCall = true; end
if firstCall && t==0
    fprintf('DEBUG-ODE: length(y) = %d\n', numel(y));
    if isstruct(parameters)
        try
            disp('Field names in param struct:');  disp(fieldnames(parameters)');
        catch
        end
    end
    firstCall = false;
end

% ===== 1. length =====
n = parameters.n;
if isempty(n) || ~isfinite(n), n = 1; end
n = max(1, round(double(n)));

% check K
if isfield(parameters,'K')
    K = parameters.K;
elseif isfield(parameters,'burst_size')
    bs = parameters.burst_size;
    if isvector(bs)
        K = numel(bs);
    else
        K = size(bs,2);
    end
else
    K = 1;
end
Ny = numel(y);

K_from_y = (Ny - 3*n - 1) / (2*n + 1);
if abs(K_from_y - round(K_from_y)) < 1e-9 && K_from_y > 0
    if 3*n + (2*n+1)*round(K_from_y) + 1 == Ny
        K = round(K_from_y);
    end
end
K = max(1, K);

if ~isfield(parameters,'e') || isempty(parameters.e)
    parameters.e = zeros(n,K);
elseif isscalar(parameters.e)
    parameters.e = parameters.e * ones(n,K);
elseif isvector(parameters.e) && numel(parameters.e)==K
    parameters.e = repmat(parameters.e(:).', n, 1);
elseif ~isequal(size(parameters.e), [n K])
    error('parameters.e must be scalar, 1×K, or n×K.');
end

% ===== 2. extract values =====
idx = 0;
Abi        = y(idx+1:idx+n);                  idx = idx + n;
nonAbi     = y(idx+1:idx+n);                  idx = idx + n;
Resistance = y(idx+1:idx+n);                  idx = idx + n;

% L_Abi: n x K
L_Abi      = reshape(y(idx+1:idx+n*K), [n, K]);   idx = idx + n*K;
% L_nonAbi: n x K
L_nonAbi   = reshape(y(idx+1:idx+n*K), [n, K]);   idx = idx + n*K;

% P: 1 x K
phage      = y(idx+1:idx+K).';                 idx = idx + K;
Q          = y(idx+1);

% constrains
Abi(Abi<1e-12)=0; nonAbi(nonAbi<1e-12)=0; Resistance(Resistance<1e-12)=0;
L_Abi(L_Abi<1e-12)=0; L_nonAbi(L_nonAbi<1e-12)=0; phage(phage<0)=0;

% ===== 3. extract parameters =====
% bacterial growth and defences
r  = parameters.r;            % n×1 
r0 = parameters.r0;           % n×1 
A  = parameters.A;            % 
c  = parameters.C;            % n×2
e  = parameters.e;
E  = parameters.E;

% mut rate
mu_Abi_to_nonAbi        = parameters.mu_Abi_to_nonAbi;
mu_nonAbi_to_Abi        = parameters.mu_nonAbi_to_Abi;
mu_Abi_to_Resistance    = parameters.mu_Abi_to_Resistance;
mu_nonAbi_to_Resistance = parameters.mu_nonAbi_to_Resistance;

% phage parameters
adsorption_rate = parameters.adsorption_rate;
burst_size      = parameters.burst_size;            
E               = parameters.E;                     
Tt              = parameters.Tt;                    
if any(~isfinite(Tt) | Tt<=0), Tt(~isfinite(Tt) | Tt<=0) = 1; end

% lag phase parameter Q
kQ = parameters.kQ;

if isscalar(r),  r  = r*ones(n,1);  end
if isscalar(r0), r0 = r0*ones(n,1); end
if isscalar(c),  c  = [c c];        end

% adsorption_rate， burst_size， E, Tt -> n×K
phage_variables = {adsorption_rate,  burst_size, E, Tt};
for i = 1:numel(phage_variables)
    if isscalar(phage_variables{i})
        phage_variables{i} = phage_variables{i} * ones(n, K);
    elseif isvector(phage_variables{i}) && numel(phage_variables{i})==K
        phage_variables{i} = repmat(phage_variables{i}(:).', [n,1]); 
    elseif ~isequal(size(phage_variables{i}), [n K])
        error('phage_variables{i} must be scalar, 1xK, or n×K.');
    end
end

adsorption_rate = phage_variables{1};
burst_size      = phage_variables{2};            
E               = phage_variables{3};                     
Tt              = phage_variables{4}; 


% ===== 4. interaction and growth =====
S_live = Abi + nonAbi + Resistance + sum(L_Abi,2) + sum(L_nonAbi,2);  % n×1
alpha  = Q; % ./ (1 + Q);           %  0..1
% g      = alpha .* r - (A * S_live);  % n×1
S_live = sum(S_live,"all"); % one total
g = alpha .* r - A * S_live;  % n×1
gAbi  = (g - c(:,1));
gNon  =  g;
gRes  = (g - c(:,2));
gAbi(gAbi<0) = 0;
gNon(gNon<0) = 0;
gRes(gRes<0) = 0;

r0_lag = r0 .* alpha;            % n×1

% ===== 5. mut rate） =====
gAbir0  = gAbi + r0_lag;
gNonr0  = gNon + r0_lag;
gResr0  = gRes + r0_lag;

mut_Abi_to_nonAbi        = gAbir0  .* Abi    .* mu_Abi_to_nonAbi;
mut_nonAbi_to_Abi        = gNonr0  .* nonAbi .* mu_nonAbi_to_Abi;
mut_Abi_to_Resistance    = gAbir0  .* Abi    .* mu_Abi_to_Resistance;
mut_nonAbi_to_Resistance = gNonr0  .* nonAbi .* mu_nonAbi_to_Resistance;

% ===== 6. infection and latent =====
% each phage k
% infection_loss_Abi(:,k) = a(:,k) * Pk .* Abi
P_row = phage(:).';                          % 1×K
infection_loss_Abi    = adsorption_rate .* (Abi    * P_row);   % n×K
infection_loss_nonAbi = adsorption_rate .* (nonAbi * P_row);   % n×K

% latent end
latent_release_Abi    = L_Abi    ./ Tt;     % n×K（按列除）
latent_release_nonAbi = (1-parameters.e) .* L_nonAbi ./ Tt;     % n×K

% ===== 7. bacteria dB/dt =====
dAbidt = gAbir0 .* Abi ...
         - mut_Abi_to_Resistance ...
         - sum(infection_loss_Abi, 2);

dnonAbidt = gNonr0 .* nonAbi ...
            - mut_nonAbi_to_Resistance ...
            - sum(infection_loss_nonAbi, 2) + sum((parameters.e .* L_nonAbi ./ Tt),2) ;

dResistancedt = gResr0 .* Resistance ...
                + mut_Abi_to_Resistance + mut_nonAbi_to_Resistance;

% dLatent
dL_Abidt    = infection_loss_Abi    - latent_release_Abi;      % n×K
dL_nonAbidt = infection_loss_nonAbi - parameters.e .* L_nonAbi ./ Tt - latent_release_nonAbi;   % n×K

% Q dynamic
dQdt = kQ * Q * (1 - Q);     % logistic

% ===== 8. phage k dyamic =====
dphagedt = zeros(1,K);
for k = 1:K
    % absorption loss
    ads_loss_nonAbi_k = sum(adsorption_rate(:,k) .* phage(k) .* nonAbi);
    ads_loss_Abi_k    = sum(adsorption_rate(:,k) .* phage(k) .* Abi);

    % phage amplification
    prod_nonAbi_k = burst_size(:,k) .* (1 - e(:,k)) .* latent_release_nonAbi(:,k);
    prod_Abi_k    = burst_size(:,k) .* (1 - E(:,k)) .* latent_release_Abi(:,k);
    production_k  = sum(prod_nonAbi_k) + sum(prod_Abi_k);

    dphagedt(k) = production_k - (ads_loss_nonAbi_k + ads_loss_Abi_k);
    
    % constrains
    if phage(k) < 1e-12 && dphagedt(k) < 0
        dphagedt(k) = 0;
    end
end

% ===== 9. return =====
dydt = [
    dAbidt;
    dnonAbidt;
    dResistancedt;
    dL_Abidt(:);
    dL_nonAbidt(:);
    dphagedt.';
    dQdt
];
end
