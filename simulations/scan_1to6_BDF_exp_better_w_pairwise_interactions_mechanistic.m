close all; clc; clear

addpath("../../model")
addpath("../../configs")

%% ==== 0) parameters ====
load('randn_bac_community_BDF_exp_param.mat');  % with parameters
parameters.A = mean(parameters.A,"all");
parameters.A = repmat(parameters.A, parameters.n, parameters.n);
parameters.mu_Abi_to_Resistance = 1e-5;
parameters.mu_nonAbi_to_Resistance = 1e-5;
parameters.r0 = repmat(mean(parameters.r0,"all"),size(parameters.r0,1),size(parameters.r0,2));

parameters.e = parameters.E;

% Order:
% 1 = PAMS
% 2 = Pycsar
% 3 = CBASS
% 4 = Shango
% 5 = Shedu
% 6 = AbiD
% 
interaction_raw = [
     0       -2.561333   9.798686   5.096350  10.759128   9.680266;
    -2.561333  0        12.803849   9.784762   6.796831   4.226573;
     9.798686 12.803849  0          9.828128   2.856048   1.385548;
     5.096350  9.784762  9.828128   0          1.068189   4.981672;
    10.759128  6.796831  2.856048   1.068189   0         -2.824989;
     9.680266  4.226573  1.385548   4.981672  -2.824989   0
]; % > 0 means positive interation thus lower inhibition in A matrix

interaction_strength = 0.2;
parameters.AA = parameters.A .* interaction_strength .* interaction_raw ./ max(interaction_raw, [], "all");



%% ==== 1) scan range ====
n_list = 1:6;
K_list = 1:3;

store_per_strain = true;         
K_focus = 3;                     

if ~exist('tspan','var'); tspan = [0 24]; end

% save the curve if possible
store_curves = true;

% time grid for plotting and reduce saving storage space
use_common_time_grid = true;
t_common = linspace(tspan(1), tspan(2), 241);  % 0~24h，0.1h a point

curves_all = cell(numel(n_list), numel(K_list));

% save combo
combos_all = cell(numel(n_list), numel(K_list));  

per_strain_curves = cell(numel(n_list), 1);
per_phage_curves = cell(numel(n_list), 1);   % 新增：存 K=3 的 phage 动态

% save AUC
AUC_all = cell(numel(n_list), numel(K_list));   
NEff    = zeros(numel(n_list), numel(K_list));  

AUC_mean = nan(numel(n_list), numel(K_list));
AUC_std  = nan(numel(n_list), numel(K_list));

common = cell(max(n_list),max(K_list)); 

%% ==== 2) initial ====
TotB0 = 2.72e8;
TotP0 = 2.72e7;   

%% ==== 3) ODE setting ====
base_opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

%% ==== 4) run ====
f_handle = @glv_continuous_no_resource_v5_latent_lag_3phage_mat_open_mecitr; 


for ii = 1:numel(n_list)
    n_now = n_list(ii);
    strain_combos = nchoosek(1:6, n_now);

    for jj = 1:numel(K_list)

        K_now = K_list(jj);

        % initialisation
        if store_per_strain && (K_now == K_focus)
            per_strain_curves{ii} = [];
        end

        if store_per_strain && (K_now == K_focus) && isempty(per_phage_curves{ii})
            per_phage_curves{ii} = [];
        end

        if store_curves
            curves_all{ii,jj} = [];   % struct
            combos_all{ii,jj} = [];
        end

        K_now = K_list(jj);
        phage_combos = nchoosek(1:K_pool, K_now);

        auc_all = nan(size(strain_combos,1) * size(phage_combos,1), 1);
        c = 0;

        AUC_all{ii,jj} = nan(size(strain_combos,1) * size(phage_combos,1), 1);

        S = struct();    % to save later in "common"
        S.cond = struct('ii',ii,'jj',jj,'parameters',parameters); 

        runs = struct('sp',{},'ph',{},'t',{},'sol',{});

        run = 0;

        for a = 1:size(strain_combos,1)
            s_idx = strain_combos(a,:);

            for b = 1:size(phage_combos,1)
                p_idx = phage_combos(b,:);
                c = c + 1;
                
                % to save in common
                sp = s_idx;     % e.g. vector of species IDs
                ph = p_idx;     % e.g. vector of phage IDs

                sol = struct();
                run = run + 1;

                % 4.1 subset of parameters
                parameters_sub = subset_parameters(parameters, s_idx, p_idx);

                % 4.2 construct y0: sum of totalB or P always the same
                Abi0       = zeros(n, 1);
                nonAbi0    = zeros(n, 1);
                Resistant0 = zeros(n, 1);
                Abi0    = (TotB0 / n_now) .* is_Abi_non_Abi_DS(:);
                nonAbi0 = (TotB0 / n_now) .* (1 - is_Abi_non_Abi_DS(:));
                
                Abi0    = Abi0(s_idx);      
                nonAbi0 = nonAbi0(s_idx);   

                Resistant0 = zeros(n_now,1);

                L_Abi0     = zeros(n_now, K_now);
                L_nonAbi0  = zeros(n_now, K_now);

                P0         = (TotP0 / K_now) * ones(K_now,1);
                Q0         = 1.0e-3;

                y0_now = [Abi0(:); nonAbi0(:); Resistant0(:); L_Abi0(:); L_nonAbi0(:); P0; Q0];

                opts_now = odeset(base_opts, 'NonNegative', 1:numel(y0_now));

                % 4.3 ode solver + AUC
                try
                    f = @(t,y) f_handle(t, y, parameters_sub);
                    [tsol, ysol] = ode15s(f, tspan, y0_now, opts_now);

                    %  -> Total_live
                    idx = 0;
                    Abi_idx        = idx + (1:n_now);       idx = idx + n_now;
                    nonAbi_idx     = idx + (1:n_now);       idx = idx + n_now;
                    Resistant_idx  = idx + (1:n_now);       idx = idx + n_now;
                    L_Abi_idx      = idx + (1:n_now*K_now); idx = idx + n_now*K_now;
                    L_nonAbi_idx   = idx + (1:n_now*K_now); idx = idx + n_now*K_now;
                    P_idx          = idx + (1:K_now);       idx = idx + K_now;
                    Q_idx          = idx + 1;
                    
                    P_sol = ysol(:, P_idx);   % [T × K_now]

                    Abi_sol        = ysol(:, Abi_idx);
                    nonAbi_sol     = ysol(:, nonAbi_idx);
                    Resistant_sol  = ysol(:, Resistant_idx);
                    L_Abi_sol      = ysol(:, L_Abi_idx);
                    L_nonAbi_sol   = ysol(:, L_nonAbi_idx);

                    T = size(ysol,1);
                    L_Abi_sum    = squeeze(sum(reshape(L_Abi_sol,    T, n_now, K_now), 3));
                    L_nonAbi_sum = squeeze(sum(reshape(L_nonAbi_sol, T, n_now, K_now), 3));

                    Total_live = Abi_sol + nonAbi_sol + Resistant_sol + L_Abi_sum + L_nonAbi_sum;

                    % save into the "common" variable
                    sol.Abi_sol       = Abi_sol;
                    sol.nonAbi_sol    = nonAbi_sol;
                    sol.Resistant_sol = Resistant_sol;
                    sol.L_Abi_sol     = L_Abi_sol;
                    sol.L_nonAbi_sol  = L_nonAbi_sol;
                    sol.P_sol         = P_sol;

                    runs(run).a_id = a;        % which row of strain_combos
                    runs(run).b_id = b;        % which row of phage_combos
                    runs(run).sp   = s_idx;    % actual strain indices used
                    runs(run).ph   = p_idx;    % actual phage indices used
                    runs(run).t   = t_common;   
                    runs(run).sol = sol;

                    % save per strain dynamic
                    if store_per_strain && (K_now == K_focus) && use_common_time_grid
                        % Total_live: [T × n_now]
                        Y_common = interp1(tsol, Total_live, t_common, 'pchip', 'extrap'); % [T_common × n_now]
                    
                        per_strain_curves{ii}(end+1,1).t         = t_common(:);
                        per_strain_curves{ii}(end,1).strain_idx  = s_idx;
                        per_strain_curves{ii}(end,1).phage_idx   = p_idx;
                        per_strain_curves{ii}(end,1).Y           = Y_common;
                    end

                    % save phage dynamic
                    if store_per_strain && (K_now == K_focus) && use_common_time_grid
                        P_common = interp1(tsol, P_sol, t_common, 'pchip', 'extrap');   % [T_common × K_now]
                    
                        per_phage_curves{ii}(end+1,1).t         = t_common(:);
                        per_phage_curves{ii}(end,1).strain_idx  = s_idx;
                        per_phage_curves{ii}(end,1).phage_idx   = p_idx;        % 全局phage编号(例如[1 2 3]代表PNM/PA5P2/phiKZ)
                        per_phage_curves{ii}(end,1).P           = P_common;     % 每列对应该次组合中的一个phage
                    end

                    auc_all(c) = trapz(tsol, log10(sum(Total_live,2)));

                    AUC_all{ii,jj}(c) = auc_all(c);

                    if store_curves
                        TotB = sum(Total_live, 2);  % 你的 Total_live 总细菌（含latent）
                    
                        if use_common_time_grid
                            % get and save time grid
                            TotB_common = interp1(tsol, TotB, t_common, 'pchip', 'extrap');
                                                        
                            curves_all{ii,jj}(end+1,1).TotB = TotB_common(:);
                            curves_all{ii,jj}(end,1).AUC    = auc_all(c);

                            ysol_list = {'Abi_sol','nonAbi_sol','Resistant_sol', ...
                                         'L_Abi_sol','L_nonAbi_sol','P_sol'};
                            
                            for i = 1:numel(ysol_list)
                                disp(i);
                                name = ysol_list{i};
                                disp(name);
                                temp = interp1(tsol, eval(name), t_common, 'pchip', 'extrap');
                                runs(run).sol.(name) = temp;
                            end

                        else
                            % save tsol?
                            curves_all{ii,jj}(end+1,1).tsol = tsol;
                            curves_all{ii,jj}(end,1).TotB  = TotB;
                            curves_all{ii,jj}(end,1).AUC   = auc_all(c);
                        end
                    
                        combos_all{ii,jj}(end+1,1).strain_idx = s_idx;
                        combos_all{ii,jj}(end,1).phage_idx    = p_idx;
                    end

                    NEff(ii,jj) = sum(~isnan(AUC_all{ii,jj}));
                    AUC_mean(ii,jj) = mean(AUC_all{ii,jj}, 'omitnan');
                    AUC_std(ii,jj)  = std(AUC_all{ii,jj},  'omitnan');

                catch
                    auc_all(c) = NaN;
                end
            end
        end

        AUC_mean(ii,jj) = mean(auc_all, 'omitnan');
        AUC_std(ii,jj)  = std(auc_all,  'omitnan');

        S.runs = runs;
        common{ii,jj} = S;

        fprintf('n=%d K=%d | mean=%.4f std=%.4f | sims=%d\n', ...
            n_now, K_now, AUC_mean(ii,jj), AUC_std(ii,jj), numel(auc_all));
    end
end

%% ==== 5) plot ====
figure('Color','w');
imagesc(K_list, n_list, AUC_mean);
set(gca,'YDir','normal');
xlabel('Phage diversity K'); ylabel('Bacteria diversity n');
title('Combinatorial mean AUC');
colorbar; grid on;

figure('Color','w');
imagesc(K_list, n_list, AUC_std);
set(gca,'YDir','normal');
xlabel('Phage diversity K'); ylabel('Bacteria diversity n');
title('Combinatorial std(AUC)');
colorbar; grid on;

save('scan_nK_AUC_combinatorial.mat', ...
     'n_list','K_list','AUC_mean','AUC_std','TotB0','TotP0', ...
     'NEff','AUC_all',...
     'store_curves','use_common_time_grid','t_common', ...
     'curves_all','combos_all', ...
     'store_per_strain','K_focus','per_strain_curves', 'per_phage_curves', ...
     'common', ...
     '-v7.3');

%% =======================================================================
function p2 = subset_parameters(p, s_idx, p_idx)
    p2 = p;
    n2 = numel(s_idx);
    K2 = numel(p_idx);

    if isfield(p2,'n'); p2.n = n2; end
    if isfield(p2,'K'); p2.K = K2; end

    % r, r0: n×1
    if isfield(p2,'r');  p2.r  = p2.r(s_idx);  end
    if isfield(p2,'r0'); p2.r0 = p2.r0(s_idx); end

    % A:  n×1 or n×n now
    if isfield(p2,'A')
        A = p2.A;
        if isvector(A)
            p2.A = A(s_idx);
        else
            p2.A = A(s_idx, s_idx);
        end
    end

    % AA
    if isfield(p2,'AA')
        AA = p2.AA;
        if isvector(AA)
            p2.AA = AA(s_idx);
        else
            p2.AA = AA(s_idx, s_idx);
        end
    end

    % C: n×2
    if isfield(p2,'C')
        p2.C = p2.C(s_idx, :);
    end

    % adsorption_rate: n×K
    if isfield(p2,'adsorption_rate')
        p2.adsorption_rate = p2.adsorption_rate(s_idx, p_idx);
    end

    % burst_size / E / Tt: 1×K or n×K
    slice_phage = @(X) ( isvector(X) * X(p_idx) + ~isvector(X) * X(s_idx, p_idx) );

    if isfield(p2,'burst_size')
        bs = p2.burst_size;
        if isvector(bs), p2.burst_size = bs(p_idx); else, p2.burst_size = bs(s_idx, p_idx); end
    end

    if isfield(p2,'E')
        E = p2.E;
        if isvector(E), p2.E = E(p_idx); else, p2.E = E(s_idx, p_idx); end
    end

    if isfield(p2,'Tt')
        Tt = p2.Tt;
        if isvector(Tt), p2.Tt = Tt(p_idx); else, p2.Tt = Tt(s_idx, p_idx); end
    end

    % e
    if isfield(p2,'e')
        e = p2.e;
        if isvector(e), p2.e = e(p_idx); else, p2.e = e(s_idx, p_idx); end
    end

end