%% Simulate community dynamics for all generated interaction matrices (PARFOR version)
% For each interaction matrix:
%   - sample host communities of size 1:100 
%   - n_rep_per_n random communities per host-richness level
%   - simulate dynamics with the same ODE model
%   - save each community time series as one worksheet in Excel
%
% Output:
%   simulation_results_whole_range/
%       k02/
%           k02_m001_simulation_results.xlsx
%           ...
%       k03/
%           ...
%
% Each workbook:
%   - one sheet per community, e.g. n10_r01, n30_r03, etc.

close all; clc; clear;

n_workers = max(feature('numcores') - 4, 1);

%% =======================
% 0) Paths & settings
%% =======================
project_root = '/mnt/fls01-home01/h63351ft/scratch/BDF_simulations';
addpath(genpath(fullfile(project_root, 'model')));
addpath(genpath(fullfile(project_root, 'lib')));
addpath(genpath(fullfile(project_root, 'configs')));

param_mat_path = fullfile(project_root, 'configs', 'BDF_exp_perturbed_resample_param.mat');
S = load(param_mat_path);              % expects 'parameters' inside
parameters = S.parameters;

% ---- input excel files ----
directory        = fullfile(project_root, 'experiments', 'BDF_small_ctrl_distribution');
out_mats_xlsx    = fullfile(directory, 'interaction_matrices_ctl_distribution.xlsx');
out_metrics_xlsx = fullfile(directory, 'network_metrics_ctl_distribution.xlsx');
phage_par_xlsx   = fullfile(directory, 'phage_param_pool_100.xlsx');

% ---- output folder ----
sim_out_dir = fullfile(directory, 'simulation_results_large_scale_partitioning/');
if ~exist(sim_out_dir, 'dir')
    mkdir(sim_out_dir);
end

% ---- simulation options ----
tspan    = [0 24];
t_common = linspace(tspan(1), tspan(2), 24*6 + 1);   % 10 min interval
base_opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

% ---- community sampling ----
% Original:
% n_list = 1:6;
% n_rep_per_n = 12;

n_list = [1 2 4 5 10 20 25 50 100];   % partitioning hosts
n_rep_per_n = 6;       % % number of random partition rounds per richness level

% ---- initial totals ----
TotB0 = 2.72e8;
TotP0 = 2.72e7;
Q0    = 1e-3;

% ---- ODE handle ----
f_handle = @glv_continuous_no_resource_v5_latent_lag_3phage_matrix_open_sum;

% ---- prototype Abi/non-Abi definition from original 6-strain setup ----
% 0 = non-Abi, 1 = Abi
proto_is_Abi_non_Abi_DS = [0 0 1 1 0 1]'; % half-half
% proto_is_Abi_non_Abi_DS = [0 0 0 0 0 0]';  % all non-Abi

% ---- phage parameter table ----
ph_par = readtable(phage_par_xlsx, 'Sheet', 'meta');

% reproducible sampling
rng(1);

%% =======================
% 1) Read metrics summary
%% =======================
metric_sheets = sheetnames(out_metrics_xlsx);
T = readtable(out_metrics_xlsx, 'Sheet', metric_sheets{1});

% Support either old or new column names
if any(strcmp(T.Properties.VariableNames, 'matrix_id'))
    sheet_ids = T.matrix_id;
elseif any(strcmp(T.Properties.VariableNames, 'sheet_id'))
    sheet_ids = T.sheet_id;
else
    error('Could not find matrix_id or sheet_id in %s', out_metrics_xlsx);
end

if any(strcmp(T.Properties.VariableNames, 'k_phage'))
    K_vals = T.k_phage;
elseif any(strcmp(T.Properties.VariableNames, 'k_cols'))
    K_vals = T.k_cols;
else
    K_vals = nan(height(T),1);
end

%% ---- K range ----
K_min = 2;
K_max = 6;

keep_idx = (K_vals >= K_min) & (K_vals <= K_max);

sheet_ids = sheet_ids(keep_idx);
K_vals    = K_vals(keep_idx);

n_cond = numel(sheet_ids);

fprintf('PARFOR MODE: running matrices with K between %d and %d\n', K_min, K_max);
fprintf('Matrices selected: %d\n', n_cond);
fprintf('n_list = [%s]\n', num2str(n_list));
fprintf('n_rep_per_n = %d\n', n_rep_per_n);

%% =======================
% 2) Start parallel pool
%% =======================
pool = gcp('nocreate');

n_workers = max(feature('numcores') - 2, 1); % leave 2 cores free

if isempty(pool)
    parpool(n_workers);
elseif pool.NumWorkers ~= n_workers
    delete(pool);
    parpool(n_workers);
end

%% =======================
% 3) Run matrices in parallel
%% =======================
parfor c = 1:n_cond
    try
        run_one_matrix_parfor( ...
            c, sheet_ids, K_vals, ...
            out_mats_xlsx, sim_out_dir, ...
            parameters, ph_par, proto_is_Abi_non_Abi_DS, ...
            tspan, t_common, base_opts, ...
            n_list, n_rep_per_n, ...
            TotB0, TotP0, Q0, ...
            f_handle);
    catch ME
        warning('Matrix %d failed in PARFOR: %s', c, ME.message);
    end
end

disp('All simulations finished.');

%% =======================================================================
function run_one_matrix_parfor( ...
    c, sheet_ids, K_vals, ...
    out_mats_xlsx, sim_out_dir, ...
    parameters, ph_par, proto_is_Abi_non_Abi_DS, ...
    tspan, t_common, base_opts, ...
    n_list, n_rep_per_n, ...
    TotB0, TotP0, Q0, ...
    f_handle)

    %% ---- get sheet name ----
    if iscell(sheet_ids)
        sh = sheet_ids{c};
    else
        sh = sheet_ids(c);
        if isstring(sh)
            sh = char(sh);
        end
    end

    fprintf('\n[Worker] Running matrix %d: %s\n', c, sh);

    %% ---- read interaction matrix ----
    Tmat = readtable(out_mats_xlsx, 'Sheet', sh, 'ReadRowNames', true);
    W = table2array(Tmat);

    defence_names = Tmat.Properties.RowNames;
    phage_ids     = Tmat.Properties.VariableNames;

    W(~isfinite(W)) = 0;
    W = max(min(W, 1), 0);

    [n_pool, K_input] = size(W);

    if isnan(K_vals(c))
        K_this = K_input;
    else
        K_this = K_vals(c);
    end

    fprintf('[Worker] %s | n_pool = %d hosts, K_input = %d phages\n', sh, n_pool, K_input);

    %% ---- build output workbook path ----
    k_folder = fullfile(sim_out_dir, sprintf('k%02d', K_input));
    if ~exist(k_folder, 'dir')
        mkdir(k_folder);
    end

    excel_out = fullfile(k_folder, sprintf('%s_simulation_results.xlsx', sh));
    if exist(excel_out, 'file')
        delete(excel_out);
    end

    %% ---- build parameters for this matrix ----
    p = parameters;

    p.A = mean(p.A, "all");

    % expand all host-dimension parameters from prototype host pool to n_pool
    p = expand_host_parameters_from_prototype(p, n_pool);

    p.n = n_pool;
    p.K = K_input;

    % fixed phage life-history traits: repeat first phage param row
    ph_par_subset = ph_par(1,:);
    ph_par_subset = ph_par_subset(repelem(1, K_input), :);

    p.adsorption_rate = repmat(ph_par_subset.adsorption_rate(:).', n_pool, K_input);
    p.burst_size      = repmat(ph_par_subset.burst_size(:).',      n_pool, K_input);
    p.Tt              = repmat(ph_par_subset.Tt(:).',              n_pool, K_input);

    % interaction matrix
    p.E = W;
    p.e = W;

    ph_idx = 1:K_input;

    % expand Abi/non-Abi labels to match n_pool
    is_Abi_non_Abi_DS = expand_binary_prototype(proto_is_Abi_non_Abi_DS, n_pool);

    %% ---- estimate max number of communities ----
    n_total_max = 0;
    for ii = 1:numel(n_list)
        n_now = n_list(ii);
        if n_now <= n_pool
            n_total_max = n_total_max + floor(n_pool / n_now) * n_rep_per_n;
        end
    end

    meta_rows    = cell(n_total_max, 8);
    sheet_tables = cell(n_total_max, 1);
    sheet_names  = cell(n_total_max, 1);

    meta_counter = 0;

    %% ---- loop over host richness ----
    for ii = 1:numel(n_list)
        n_now = n_list(ii);

        if n_now > n_pool
            warning('%s: n_now=%d exceeds n_pool=%d. Skipping.', sh, n_now, n_pool);
            continue;
        end

        [comms, rep_id_within_partition] = sample_random_partitioned_communities(n_pool, n_now, n_rep_per_n);
        
        for rr = 1:size(comms,1)
            meta_counter = meta_counter + 1;

            s_idx = comms(rr,:);
            sheet_name = sprintf('n%d_c%03d', n_now, rr);

            p_sub = subset_parameters(p, s_idx, ph_idx);

            % initial conditions
            Abi0    = (TotB0 / n_now) .* is_Abi_non_Abi_DS(:);
            nonAbi0 = (TotB0 / n_now) .* (1 - is_Abi_non_Abi_DS(:));

            Abi0    = Abi0(s_idx);
            nonAbi0 = nonAbi0(s_idx);

            Resistant0 = zeros(n_now,1);
            L_Abi0     = zeros(n_now, K_input);
            L_nonAbi0  = zeros(n_now, K_input);

            P0 = (TotP0 / K_input) * ones(K_input,1);

            y0_now = [Abi0(:); nonAbi0(:); Resistant0(:); L_Abi0(:); L_nonAbi0(:); P0; Q0];
            opts_now = odeset(base_opts, 'NonNegative', 1:numel(y0_now));

            % host names for metadata
            this_host_names = defence_names(s_idx);
            if isstring(this_host_names)
                this_host_names = cellstr(this_host_names);
            end
            if ischar(this_host_names)
                this_host_names = {this_host_names};
            end

            status_txt = "success";

            try
                f = @(t,y) f_handle(t, y, p_sub);
                [tsol, ysol] = ode15s(f, tspan, y0_now, opts_now);

                Tout = build_output_table_from_solution( ...
                    tsol, ysol, t_common, ...
                    n_now, K_input, ...
                    this_host_names, phage_ids);

            catch ME
                status_txt = "failed";
                warning('Simulation failed for %s | %s: %s', sh, sheet_name, ME.message);

                % minimal output sheet if failed
                Tout = table(t_common(:), 'VariableNames', {'time_h'});
                Tout.TotalB = nan(numel(t_common),1);
                Tout.TotalP = nan(numel(t_common),1);
                Tout.Q      = nan(numel(t_common),1);
            end

            % store for later writing
            sheet_tables{meta_counter} = Tout;
            sheet_names{meta_counter}  = sheet_name;

            meta_rows(meta_counter, :) = { ...
                sheet_name, ...
                sh, ...
                K_this, ...
                n_now, ...
                rep_id_within_partition(rr), ...
                strjoin(string(s_idx), ';'), ...
                strjoin(string(this_host_names), ';'), ...
                status_txt ...
                };
        end
    end

    %% ---- trim unused rows ----
    meta_rows    = meta_rows(1:meta_counter, :);
    sheet_tables = sheet_tables(1:meta_counter);
    sheet_names  = sheet_names(1:meta_counter);

    %% ---- write output workbook within this worker ----
    for kk = 1:numel(sheet_names)
        writetable(sheet_tables{kk}, excel_out, 'Sheet', sheet_names{kk});
    end

    meta_tbl = cell2table(meta_rows, 'VariableNames', ...
        {'sheet_name','matrix_id','K_phage','n_hosts','replicate','host_indices','host_names','status'});

    writetable(meta_tbl, excel_out, 'Sheet', 'community_meta');

    fprintf('[Worker] Saved: %s\n', excel_out);
end

%% =======================================================================
function Tout = build_output_table_from_solution(tsol, ysol, t_common, n_now, K_input, this_host_names, phage_ids)

    % ---- unpack variables ----
    idx = 0;
    Abi_idx        = idx + (1:n_now);          idx = idx + n_now;
    nonAbi_idx     = idx + (1:n_now);          idx = idx + n_now;
    Resistant_idx  = idx + (1:n_now);          idx = idx + n_now;
    L_Abi_idx      = idx + (1:n_now*K_input);  idx = idx + n_now*K_input;
    L_nonAbi_idx   = idx + (1:n_now*K_input);  idx = idx + n_now*K_input;
    P_idx          = idx + (1:K_input);        idx = idx + K_input;
    Q_idx          = idx + 1;

    Abi_sol       = ysol(:, Abi_idx);
    nonAbi_sol    = ysol(:, nonAbi_idx);
    Resistant_sol = ysol(:, Resistant_idx);
    L_Abi_sol     = ysol(:, L_Abi_idx);
    L_nonAbi_sol  = ysol(:, L_nonAbi_idx);
    P_sol         = ysol(:, P_idx);
    Q_sol         = ysol(:, Q_idx);

    Tn = size(ysol,1);

    L_Abi_arr    = reshape(L_Abi_sol,    Tn, n_now, K_input);
    L_nonAbi_arr = reshape(L_nonAbi_sol, Tn, n_now, K_input);

    L_Abi_sum    = reshape(sum(L_Abi_arr,    3), Tn, n_now);
    L_nonAbi_sum = reshape(sum(L_nonAbi_arr, 3), Tn, n_now);

    Total_live = Abi_sol + nonAbi_sol + Resistant_sol + L_Abi_sum + L_nonAbi_sum;
    TotB = sum(Total_live, 2);
    TotP = sum(P_sol, 2);

    % ---- interpolate to common time grid ----
    Abi_i        = interp1(tsol, Abi_sol,       t_common, 'pchip', 'extrap');
    nonAbi_i     = interp1(tsol, nonAbi_sol,    t_common, 'pchip', 'extrap');
    Resistant_i  = interp1(tsol, Resistant_sol, t_common, 'pchip', 'extrap');
    L_Abi_sum_i  = interp1(tsol, L_Abi_sum,     t_common, 'pchip', 'extrap');
    L_nonAbi_i   = interp1(tsol, L_nonAbi_sum,  t_common, 'pchip', 'extrap');
    Total_live_i = interp1(tsol, Total_live,    t_common, 'pchip', 'extrap');
    P_i          = interp1(tsol, P_sol,         t_common, 'pchip', 'extrap');
    TotB_i       = interp1(tsol, TotB,          t_common, 'pchip', 'extrap');
    TotP_i       = interp1(tsol, TotP,          t_common, 'pchip', 'extrap');
    Q_i          = interp1(tsol, Q_sol,         t_common, 'pchip', 'extrap');

    % enforce expected shapes
    Abi_i        = reshape(Abi_i,        numel(t_common), n_now);
    nonAbi_i     = reshape(nonAbi_i,     numel(t_common), n_now);
    Resistant_i  = reshape(Resistant_i,  numel(t_common), n_now);
    L_Abi_sum_i  = reshape(L_Abi_sum_i,  numel(t_common), n_now);
    L_nonAbi_i   = reshape(L_nonAbi_i,   numel(t_common), n_now);
    Total_live_i = reshape(Total_live_i, numel(t_common), n_now);
    P_i          = reshape(P_i,          numel(t_common), K_input);

    TotB_i = TotB_i(:);
    TotP_i = TotP_i(:);
    Q_i    = Q_i(:);

    % ---- build output table ----
    Tout = table(t_common(:), 'VariableNames', {'time_h'});

    for h = 1:n_now
        host_tag = matlab.lang.makeValidName(sprintf('h%02d_%s', h, this_host_names{h}));

        Tout.(sprintf('Abi_%s', host_tag))        = Abi_i(:,h);
        Tout.(sprintf('nonAbi_%s', host_tag))     = nonAbi_i(:,h);
        Tout.(sprintf('Resistant_%s', host_tag))  = Resistant_i(:,h);
        Tout.(sprintf('LAbiSum_%s', host_tag))    = L_Abi_sum_i(:,h);
        Tout.(sprintf('LnonAbiSum_%s', host_tag)) = L_nonAbi_i(:,h);
        Tout.(sprintf('TotalLive_%s', host_tag))  = Total_live_i(:,h);
    end

    for pp = 1:K_input
        phage_tag = matlab.lang.makeValidName(sprintf('p%02d_%s', pp, phage_ids{pp}));
        Tout.(sprintf('Phage_%s', phage_tag)) = P_i(:,pp);
    end

    Tout.TotalB = TotB_i;
    Tout.TotalP = TotP_i;
    Tout.Q      = Q_i;
end

%% =======================================================================
function [comms, rep_id] = sample_random_partitioned_communities(n_pool, n_now, n_rep)
% Randomly partition the full host pool into communities of size n_now.
% For each replicate:
%   - shuffle all hosts
%   - split into consecutive groups of size n_now
%   - discard leftover hosts if n_pool is not divisible by n_now
%
% Outputs:
%   comms  : total_communities x n_now matrix
%   rep_id : total_communities x 1 vector indicating partition replicate ID

    n_comm_per_rep = floor(n_pool / n_now);

    if n_comm_per_rep < 1
        warning('n_now=%d exceeds n_pool=%d. No communities returned.', n_now, n_pool);
        comms = zeros(0, n_now);
        rep_id = zeros(0,1);
        return;
    end

    n_total = n_comm_per_rep * n_rep;
    comms = zeros(n_total, n_now);
    rep_id = zeros(n_total, 1);

    counter = 0;

    for r = 1:n_rep
        perm = randperm(n_pool);

        usable = perm(1:(n_comm_per_rep * n_now));
        comms_r = reshape(usable, n_now, n_comm_per_rep).';
        comms_r = sort(comms_r, 2);   % optional: sort within each community for cleaner metadata

        idx = counter + (1:n_comm_per_rep);
        comms(idx, :) = comms_r;
        rep_id(idx) = r;

        counter = counter + n_comm_per_rep;
    end
end

%% ==========================================================================
function out = expand_binary_prototype(proto_vec, n_target)
% Expand a binary prototype vector by random resampling with replacement

    proto_vec = proto_vec(:);
    idx = randi(numel(proto_vec), n_target, 1);
    out = proto_vec(idx);
end

%% ========================================================================
function p2 = subset_parameters(p, s_idx, p_idx)
% Subset parameters by host indices s_idx and phage indices p_idx

    p2 = p;
    n2 = numel(s_idx);
    K2 = numel(p_idx);

    if isfield(p2,'n'); p2.n = n2; end
    if isfield(p2,'K'); p2.K = K2; end

    if isfield(p2,'r');  p2.r  = p2.r(s_idx);  end
    if isfield(p2,'r0'); p2.r0 = p2.r0(s_idx); end

    if isfield(p2,'C')
        p2.C = p2.C(s_idx, :);
    end

    if isfield(p2,'adsorption_rate')
        ar = p2.adsorption_rate;
        if isvector(ar)
            p2.adsorption_rate = ar(p_idx);
        else
            p2.adsorption_rate = ar(s_idx, p_idx);
        end
    end

    if isfield(p2,'burst_size')
        bs = p2.burst_size;
        if isvector(bs)
            p2.burst_size = bs(p_idx);
        else
            p2.burst_size = bs(s_idx, p_idx);
        end
    end

    if isfield(p2,'E')
        E = p2.E;
        if isvector(E)
            p2.E = E(p_idx);
        else
            p2.E = E(s_idx, p_idx);
        end
    end

    if isfield(p2,'Tt')
        Tt = p2.Tt;
        if isvector(Tt)
            p2.Tt = Tt(p_idx);
        else
            p2.Tt = Tt(s_idx, p_idx);
        end
    end

    if isfield(p2,'e')
        e = p2.e;
        if isvector(e)
            p2.e = e(p_idx);
        else
            p2.e = e(s_idx, p_idx);
        end
    end

    if isfield(p2,'r')  && isvector(p2.r),  p2.r  = p2.r(:);  end
    if isfield(p2,'r0') && isvector(p2.r0), p2.r0 = p2.r0(:); end
end

%% =======================================================================
function p = expand_host_parameters_from_prototype(p, n_target)
% Expand host-dimension parameters from an original prototype host pool
% to n_target hosts by resampling host rows with replacement.

    % infer prototype host number
    n_proto = [];

    if isfield(p, 'r') && ~isempty(p.r)
        n_proto = numel(p.r);
    elseif isfield(p, 'r0') && ~isempty(p.r0)
        n_proto = numel(p.r0);
    elseif isfield(p, 'C') && ~isempty(p.C)
        n_proto = size(p.C, 1);
    elseif isfield(p, 'E') && ~isempty(p.E)
        n_proto = size(p.E, 1);
    else
        error('Cannot infer prototype host number from p.');
    end

    % if already correct size, keep as is
    if n_proto == n_target
        if isfield(p,'n')
            p.n = n_target;
        end
        return
    end

    % sample prototype host identities with replacement
    host_idx = randi(n_proto, n_target, 1);

    % ---- host-level vectors ----
    if isfield(p,'r') && ~isempty(p.r)
        if isvector(p.r) && numel(p.r) == n_proto
            p.r = p.r(host_idx);
            p.r = p.r(:);
        end
    end

    if isfield(p,'r0') && ~isempty(p.r0)
        if isvector(p.r0) && numel(p.r0) == n_proto
            p.r0 = p.r0(host_idx);
            p.r0 = p.r0(:);
        end
    end

    % ---- host-level matrices ----
    if isfield(p,'C') && ~isempty(p.C)
        if size(p.C,1) == n_proto
            p.C = p.C(host_idx, :);
        end
    end

    % ---- host x phage matrices ----
    host_by_phage_fields = {'E','e','adsorption_rate','burst_size','Tt'};

    for ff = 1:numel(host_by_phage_fields)
        fld = host_by_phage_fields{ff};
        if isfield(p, fld) && ~isempty(p.(fld))
            X = p.(fld);

            if ~isvector(X) && size(X,1) == n_proto
                p.(fld) = X(host_idx, :);
            end
        end
    end

    if isfield(p,'n')
        p.n = n_target;
    end
end