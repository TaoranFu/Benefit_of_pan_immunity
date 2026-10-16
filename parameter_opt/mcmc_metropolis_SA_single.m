function mcmc_metropolis_SA_single(target_condition, data_path, Abi, save_dir)
% target_condition: 'phage-free', 'PA5P2', 'phiKZ', 'PNM'

close all

FAST_DEBUG = false;   

rng(42); 

% load in conditions
[condList, p0, lb, ub, meta, odeOpt, current_condition] = mfile_v3_single_condition_adjusted_by_trajectories(target_condition, data_path, Abi);

% 1) open parameter set
if strcmp(current_condition, 'phage-free')
    fittable = {'r','r0','A','kQ'};  % phage-free
elseif Abi == 1
    fittable = {'E'};
elseif Abi == 0.5
    fittable = {'e'};
elseif Abi == 0
    fittable = {'adsorption_rate','burst_size','mu_nonAbi_to_Resistance'};
else
    fprinf("???");
end

if isempty(fittable)
    param_best = p0;
    Ebest = 0;
    u_best = [];
    
    % save result
    timestamp = datestr(now, 'mmdd_HH_MM');
    save_filename = sprintf('sa_single_%s_%s.mat', current_condition, timestamp);
    save(save_filename, 'param_best', 'Ebest', 'current_condition', 'condList');
    return;
end

[p0_fit, lb_fit, ub_fit] = sub_pick_fields(p0, lb, ub, fittable);

[pvec0_fit, meta_fit] = struct2vec_fittable(p0_fit, lb_fit, ub_fit);
lb_vec = struct2vec_fittable(lb_fit, lb_fit, ub_fit);  
ub_vec = struct2vec_fittable(ub_fit, lb_fit, ub_fit);

% all >0
if strcmp(current_condition, 'phage-free')
    posNames = {'r','r0','A','kQ'};  % phage-free
elseif Abi == 1
    posNames = {'E'};
elseif Abi == 0.5
    posNames = {'e'};
elseif Abi == 0
    posNames = {'adsorption_rate','burst_size','mu_nonAbi_to_Resistance'};
end
isPos = make_mask(meta_fit, posNames);

log_lb = log(max(lb_vec(isPos), 1e-30));
log_ub = log(max(ub_vec(isPos), 1e-29));

to_unit   = @(x) phys2unit(x, lb_vec, ub_vec, isPos, log_lb, log_ub);
from_unit = @(u) unit2phys(u, lb_vec, ub_vec, isPos, log_lb, log_ub);

u_cur = to_unit(pvec0_fit);

% objective function（SSE）
Efun = @(u) energy_of_u_single(u, from_unit, p0, meta_fit, fittable, condList, meta, odeOpt);

[Ecur, param_cur] = Efun(u_cur);
Efirst = Ecur; Ebest = Ecur;  u_best = u_cur;  param_best = param_cur;

% global + fmincon（Hybrid），in parallel
winTrain.train_frac = 0.80;                % 训练使用前80%
winValList = [0.60 0.70 0.80];             % 多个滚动切分

Efun_train = @(u) energy_of_u_window_single(u, from_unit, p0, meta_fit, fittable, condList, meta, odeOpt, winTrain);

test_u = u_cur(:)';
test_E = Efun_train(test_u);

test_u_modified = test_u;
test_u_modified(1) = test_u_modified(1) + 0.1;  % 修改第一个参数
test_u_modified = max(0, min(1, test_u_modified));  % 确保在[0,1]范围内
test_E_modified = Efun_train(test_u_modified);

test_params = from_unit(test_u);
test_params_modified = from_unit(test_u_modified);

% SA bound
lb_u = zeros(size(u_cur));
ub_u = ones(size(u_cur));

if FAST_DEBUG
    n_starts = 2;  % ↓ 从 8 降到 2
    saOpts = optimoptions('simulannealbnd', ...
        'MaxIterations',           150, ...     
        'MaxFunctionEvaluations',  2000, ...    
        'InitialTemperature',      1, ...        
        'ReannealInterval',        50, ...
        'TemperatureFcn',          @temperatureexp, ...
        'AnnealingFcn',            @annealingfast, ...
        'FunctionTolerance',       1e-3, ...    
        'MaxStallIterations',      100, ...      
        'HybridFcn',               [], ...       
        'Display',                 'off');
else
    n_starts = 8;
    saOpts = optimoptions('simulannealbnd', ...
    'MaxIterations',           1000,       ... 
    'MaxFunctionEvaluations',  100000,      ... 
    'InitialTemperature',      50,         ... 
    'ReannealInterval',        350,         ... 
    'TemperatureFcn',          @temperatureexp,  ... 
    'AnnealingFcn',            @annealingboltz, ...
    'FunctionTolerance',       1e-5,        ... 
    'MaxStallIterations',      2000,        ... 
    'HybridFcn',               {@fmincon, optimoptions('fmincon','Display','off','Algorithm','interior-point')}, ... % 保留，这是王牌
    'OutputFcn',               @myOutputFcn, ... 
    'Display',                 'iter'); 
end

% LHS
try
    starts_u = lhsdesign(n_starts, numel(u_cur));
catch
    warning('lhsdesign not available');
    starts_u = rand(n_starts, numel(u_cur));
end
starts_u(1,:) = u_cur(:)';

try
    if isempty(gcp('nocreate'))
        parpool('local', 10);
    end
catch ME
    warning('cannot start worker：%s', ME.message);
end

best_E = inf; best_u = u_cur(:)';
all_sa = cell(n_starts,1);

parfor k = 1:n_starts
    uk0 = starts_u(k,:);

    saOptsK = saOpts;  % 先拷贝
    saOptsK = optimoptions(saOptsK, ...
   'OutputFcn', @(options,optimValues,state) myOutputFcn(options,optimValues,state,k), ...
   'Display','off');

    % run
    [uk_opt, Ek_opt] = simulannealbnd(Efun, ...
        uk0, lb_u(:)', ub_u(:)', saOptsK);
            % Efun_train, ...

    % save best
    all_sa{k} = struct('u', uk_opt, 'E', Ek_opt);
end

% read back
histCell = cell(n_starts,1);

for k = 1:n_starts
    fname = sprintf('sa_hist_run_%d.mat', k);
    if exist(fname,'file')
        S = load(fname, 'hist_to_save');
        H = S.hist_to_save;   % struct: iter, fval, bestfval, temp, x(cell)

        U = cell2mat(H.x);        % [N x P] unit space
        N = size(U,1);

        ThetaPhys = nan(N, size(U,2));
        for s = 1:N
            ThetaPhys(s,:) = from_unit(U(s,:)')';  % physical vector (row)
        end

        feasible = isfinite(H.fval) & (H.fval < 1e6);
        accepted = true(size(feasible));

        histCell{k} = struct( ...
            'theta',    ThetaPhys, ...
            'energy',   H.fval(:), ...
            'accepted', accepted(:), ...
            'feasible', feasible(:) ...
        );
    else
        histCell{k} = [];
    end
end

% ===== make vector-level paramNames =====
paramNamesVec = {};
for i = 1:numel(fittable)
    f = fittable{i};
    info = meta_fit.(f);
    if info(1) > 0
        len = info(2);
        if len == 1
            paramNamesVec{end+1} = f;
        else
            for j = 1:len
                paramNamesVec{end+1} = sprintf('%s(%d)', f, j);
            end
        end
    end
end
paramNames = paramNamesVec;

% ===== plot =====
outDir = fullfile(save_dir, 'diagnostics', 'walk_traj_phys');

opts = struct( ...
    'lineAcceptedOnly', true, ...
    'showRejected', true, ...
    'showInfeasible', true, ...
    'lb', lb_vec(:), ...
    'ub', ub_vec(:) ...
);

plot_worker_randomwalk_trajectory(histCell, paramNames, outDir, opts);

figure; hold on; grid on;
leg = {};
for k = 1:n_starts
    fname = sprintf('sa_hist_run_%d.mat', k);
    if exist(fname,'file')
        S = load(fname, 'hist_to_save');
        H = S.hist_to_save;   % struct
        % plot(H(:,1), H(:,3), '-',  'LineWidth', 1.0); % current fval
        % plot(H(:,1), H(:,2), '--', 'LineWidth', 1.0); % best fval
        plot(H.iter, H.fval, '-',  'LineWidth', 1.0);
        plot(H.iter, H.bestfval, '--', 'LineWidth', 1.0);
        leg{end+1} = sprintf('run %d cur', k);
        leg{end+1} = sprintf('run %d best', k);
    end
end
set(gca, 'YScale', 'log'); 
xlabel('Iteration'); ylabel('Objective (SSE)');
title('SA energy traces (current vs. best) across runs');
legend(leg, 'Location','bestoutside');


% best?
for k = 1:n_starts
    if ~isempty(all_sa{k}) && isfield(all_sa{k},'E')
        if all_sa{k}.E < best_E
            best_E = all_sa{k}.E;
            best_u = all_sa{k}.u;
        end
    end
end

% best!
[~, param_best] = energy_of_u_single(best_u, from_unit, p0, meta_fit, fittable, condList, meta, odeOpt, 'params_only');
Ebest = best_E; u_best = best_u(:);

% check
for i = 1:numel(fittable)
    param_name = fittable{i};
    if isfield(p0, param_name) && isfield(param_best, param_name)
        p0_val = p0.(param_name);
        best_val = param_best.(param_name);
        fprintf('%s:\n', param_name);
        fprintf('  initial: %s\n', mat2str(p0_val, 4));
        fprintf('  best: %s\n', mat2str(best_val, 4));
        if isequal(p0_val, best_val)
            fprintf('  -> havent change!!！\n');
        else
            fprintf('  -> updated\n');
        end
    end
end

%% check
for i = 1:numel(winValList)
    frac = winValList(i);
    trainW.train_frac = frac;
    valW.val_after_frac = frac;
    [Etrain, Eval] = evaluate_train_val_single(param_best, condList, meta, p0, odeOpt, trainW, valW);
    fprintf('  but %.0f%%：Train SSE=%.3g  |  Val SSE=%.3g\n', frac*100, Etrain, Eval);
end

%% report
fprintf('\n=== 优化完成 ===\n');
fprintf('最终最佳训练SSE: %.6e\n', Ebest);
fprintf('相比初始值改善: %.2f%%\n', max(0, (Efirst - Ebest) / max(Efirst, eps) * 100));

% construct
param_init = vec2struct_fittable(from_unit(u_cur), p0, meta_fit);  
param_init.K = 1;
param_init.n = 1;
plot_fit_overlay(condList{1}, param_init, param_best, odeOpt, 'Initial', 'Best');

% save
timestamp = datestr(now, 'mmdd_HH_MM');
save_filename = fullfile(save_dir,sprintf('sa_single_%s_%s.mat', current_condition, timestamp));
save(char(save_filename), ...
     'u_best','Ebest','param_best','current_condition', ...
     'fittable','meta_fit','p0','lb_vec','ub_vec','winTrain','winValList', ...
     'condList','target_condition');

fprintf('result save here: %s\n', char(save_filename));

end



% ========================================================================================
function [E, full_params] = energy_of_u_single(u, from_unit, p0, meta_fit, fittable, condList, meta, odeOpt, mode)
    vec_fit = from_unit(u);
    full_params = vec2struct_fittable(vec_fit, p0, meta_fit);
    if nargin>=9 && strcmp(mode,'params_only'), E=[]; return; end
    res = objective_fun_full_single(full_params, condList, meta, p0, odeOpt);
    E = sum(res.^2);  % SSE 能量
end

function [pfit, lfit, ufit] = sub_pick_fields(p0, lb, ub, names)
    pfit = struct(); lfit = struct(); ufit = struct();
    for i=1:numel(names)
        f = names{i};
        pfit.(f) = p0.(f);
        lfit.(f) = lb.(f);
        ufit.(f) = ub.(f);
    end
end

function [vec, meta] = struct2vec(S)
vec  = [];
meta = struct();
fields = fieldnames(S);
pos = 1;
for i = 1:numel(fields)
    f   = fields{i};
    val = S.(f);
    len = numel(val);
    vec = [vec ; val(:)];
    meta.(f) = [pos, len];
    pos = pos + len;
end
end

function S = vec2struct(vec, template, meta)
fields = fieldnames(template);
for i = 1:numel(fields)
    f    = fields{i};
    idx  = meta.(f);
    part = vec(idx(1):idx(1)+idx(2)-1);
    S.(f) = reshape(part, size(template.(f)));
end
end

function [vec, meta] = struct2vec_fittable(p0, lb, ub)
vec = [];
meta = struct(); 
pos = 1;
fields = fieldnames(p0);
for i = 1:numel(fields)
    f = fields{i};
    v0 = p0.(f); 
    L = lb.(f); 
    U = ub.(f);
    if all(isfinite(L(:))) && all(isfinite(U(:))) && any(U(:) > L(:))
        vec = [vec; v0(:)];
        meta.(f) = [pos, numel(v0)];
        pos = pos + numel(v0);
    else
        meta.(f) = [0, numel(v0)]; 
    end
end
end

function S = vec2struct_fittable(vec, template, meta)
S = template;  
fields = fieldnames(meta);  
idx = 1;
for i = 1:numel(fields)
    f = fields{i};
    info = meta.(f);
    if info(1) > 0  
        n = info(2);
        S.(f) = reshape(vec(idx:idx+n-1), size(template.(f)));
        idx = idx + n;
    end
end
end

% ========== objective_fun_full_single ==========
function res = objective_fun_full_single(param_struct, condList, meta, template, odeOpt, window)
    res  = [];
    epsB = 1e-12;

    persistent call_count_obj;
    if isempty(call_count_obj), call_count_obj = 0; end
    call_count_obj = call_count_obj + 1;

    if isfield(param_struct,'n'); n = double(param_struct.n); else; n = 1; end

    for c = 1:numel(condList)
        cond = condList{c};

        % initial：[Abi, nonAbi, Resistant, L_Abi, L_nonAbi, Phage]
        Y0 = [cond.initB_Abi, cond.initB_nonAbi, cond.Resistant, 0, 0, cond.initP, cond.Q]';

        try
            [t,sol] = ode15s(@(t,y)glv_wrapper_latent_single(t,y,param_struct,cond), cond.t, Y0, odeOpt);
            if numel(t) < 2
                diff = 1e6 * ones(numel(cond.t),1);           % if ode failed
            else
                cols = 1 : 5*max(1,n);                         % Abi+nonAbi+Resistant+LatentAbi+LatentnonAbi
                Bmodel_total = interp1(t, sum(sol(:,cols),2), cond.t, 'linear', 'extrap');

                % check valid
                valid = ~isnan(cond.Bexp) & isfinite(Bmodel_total);

                % window and check
                idx_all = true(size(cond.t));
                if nargin >= 6 && ~isempty(window)
                    if isfield(window, 'train_frac')
                        tcut = min(cond.t) + (max(cond.t) - min(cond.t)) * window.train_frac;
                        idx_all = cond.t <= tcut;
                    elseif isfield(window, 'val_after_frac')
                        tcut = min(cond.t) + (max(cond.t) - min(cond.t)) * window.val_after_frac;
                        idx_all = cond.t > tcut;
                    end
                end
                idx = valid & idx_all(:);

                % main objective
                diff = log10(Bmodel_total + epsB) - log10(cond.Bexp + epsB);
                % diff = diff(idx);                              % 只保留窗口内有效点

                % slope
                dy_model = gradient(log10(Bmodel_total + epsB), cond.t);
                dy_data  = gradient(log10(cond.Bexp + epsB),    cond.t);
            
                lambda = 0.05;                    % weighted
                res_slope = sqrt(lambda) .* (dy_model - dy_data);

                if ~exist('idx','var') || isempty(idx)
                    idx = true(size(cond.t));        
                end
                valid = isfinite(diff) & isfinite(res_slope) & idx(:);

                diff = [diff(valid) ; res_slope(valid)];

                % punishment
                alpha   = 0.10;                              
                q = prctile(cond.Bexp(valid),5,'all');
                floorObsLog = log10(alpha * max(q, epsB));   
                tooLow = valid & (log10(Bmodel_total + epsB) < floorObsLog);
                
                if any(tooLow)
                    lambda_pen = 1.0;                    
                    penResidual = sqrt(lambda_pen) .* (floorObsLog - log10(Bmodel_total(tooLow) + epsB));
                    diff = [diff ; penResidual(:)];
                end

                if any(isnan(diff)) || any(~isfinite(diff))
                    warning('Objective produced NaN/Inf at this parameter set, replaced by large penalty.');
                    diff = 1e6 * ones(numel(cond.t),1);
                end
            end
        catch ME
            warning('objective_fun_full_single failed at condition %d: %s', c, ME.message);
            diff = 1e6 * ones(numel(cond.t),1);                % ODE failed?
        end

        res = [res ; diff(:)];
    end
end

function mask = make_mask(meta, names)
fields = fieldnames(meta);
L = 0;
for i = 1:numel(fields)
    idx = meta.(fields{i});
    L = max(L, idx(1)+idx(2)-1);
end
mask = false(L,1);
for i = 1:numel(names)
    f = names{i};
    if isfield(meta, f)
        idx = meta.(f);
        mask(idx(1):idx(1)+idx(2)-1) = true;
    end
end
end

function u = phys2unit(x, lb, ub, isPos, log_lb, log_ub)
x     = x(:);          
lb    = lb(:);
ub    = ub(:);
isPos = isPos(:);

u = (x - lb) ./ max(ub - lb, eps);     
if any(isPos)
    xi = max(x(isPos), 1e-300);
    u(isPos) = (log(xi) - log_lb(:)) ./ (log_ub(:) - log_lb(:));
end
u = min(max(u,0),1);
end

function x = unit2phys(u, lb, ub, isPos, log_lb, log_ub)
u     = u(:);           
lb    = lb(:);
ub    = ub(:);
isPos = isPos(:);

x = lb + u .* (ub - lb);     
if any(isPos)
    x(isPos) = exp( log_lb(:) + u(isPos) .* (log_ub(:) - log_lb(:)) );
end
end

function E = energy_of_u_window_single(u, from_unit, p0, meta_fit, fittable, condList, meta, odeOpt, win)
    persistent call_count;
    if isempty(call_count), call_count = 0; end
    call_count = call_count + 1;

    vec_fit = from_unit(u(:));
    if call_count <= 3
        fprintf('  vec_fit dimension: %s\n', mat2str(size(vec_fit)));
        fprintf('  vec_fit value: %s\n', mat2str(vec_fit, 4));
    end
    
    full_params = vec2struct_fittable(vec_fit, p0, meta_fit);
    
    if call_count <= 3
        for i = 1:numel(fittable)
            name = fittable{i};
            if isfield(full_params, name)
                fprintf('    %s: %s\n', name, mat2str(full_params.(name), 4));
            end
        end
    end
    
    res = objective_fun_full_single(full_params, condList, meta, p0, odeOpt, win);
    E = sum(res.^2);
    
end

function [Etrain, Eval] = evaluate_train_val_single(param_struct, condList, meta, p0, odeOpt, winTrain, winVal)
    rTrain = objective_fun_full_single(param_struct, condList, meta, p0, odeOpt, winTrain);
    Etrain = sum(rTrain.^2);

    rVal   = objective_fun_full_single(param_struct, condList, meta, p0, odeOpt, winVal);
    Eval   = sum(rVal.^2);
end

function [stop, options, optchanged] = myOutputFcn(options, optimValues, state, run_id)
persistent H
stop = false; optchanged = false;

switch state
    case 'init'
        H = struct();
        H.iter = [];
        H.bestfval = [];
        H.fval = [];
        H.temp = [];
        H.x = {};        % cell array, each cell is 1xP vector (current u)
        H.bestx = {};    % may be empty

    case 'iter'
        iter = optimValues.iteration;

        % current objective
        if isfield(optimValues,'fval'), cf = optimValues.fval; else, cf = NaN; end
        % best objective
        if isfield(optimValues,'bestfval'), bf = optimValues.bestfval; else, bf = cf; end

        % temperature
        t = NaN;
        if isfield(optimValues,'temperature') && ~isempty(optimValues.temperature)
            t = optimValues.temperature(1);
        end

        % current point (u in [0,1])
        xcur = [];
        if isfield(optimValues,'x') && ~isempty(optimValues.x)
            xcur = optimValues.x(:)'; % force row
        end

        % best point (if provided)
        xbest = [];
        if isfield(optimValues,'bestx') && ~isempty(optimValues.bestx)
            xbest = optimValues.bestx(:)';
        end

        H.iter(end+1,1)    = iter;
        H.bestfval(end+1,1)= bf;
        H.fval(end+1,1)    = cf;
        H.temp(end+1,1)    = t;
        H.x{end+1,1}       = xcur;
        H.bestx{end+1,1}   = xbest;

    case 'done'
        fname = sprintf('sa_hist_run_%d.mat', run_id);
        hist_to_save = H; %#ok<NASGU>
        save(fname, 'hist_to_save');
end
end


function dY = glv_wrapper_latent_single(t, Y, param, cond) 
    dY = glv_continuous_no_resource_v5_latent_lag_3phage_matrix_open_sum(t, Y, param);
end

function plot_fit_overlay(cond, param_init, param_best, odeOpt, label_init, label_best)
if nargin < 5, label_init = 'Initial'; end
if nargin < 6, label_best = 'Best'; end

Y0 = [cond.initB_Abi, cond.initB_nonAbi, cond.Resistant, 0, 0, cond.initP, cond.Q]';

[t0,sol0] = ode15s(@(t,y)glv_continuous_no_resource_v5_latent_lag_3phage_matrix_open_sum(t, y, param_init), cond.t, Y0, odeOpt);
B0 = sum(sol0(:,1:5),2);

[t1,sol1] = ode15s(@(t,y)glv_continuous_no_resource_v5_latent_lag_3phage_matrix_open_sum(t, y, param_best), cond.t, Y0, odeOpt);
B1 = sum(sol1(:,1:5),2);

figure; hold on; grid on;

semilogy(cond.t, cond.Bexp, 'o', 'MarkerSize', 6, 'DisplayName', 'Data');
semilogy(t0, B0, '-', 'LineWidth', 1.5, 'DisplayName', label_init);
semilogy(t1, B1, '-', 'LineWidth', 1.5, 'DisplayName', label_best);
xlabel('Time'); ylabel('Total bacteria (log scale)'); title('Data vs. model');
legend('Location','best'); 
% Set only the upper limit (lower stays automatic)
ylim([0 1e10]);
end
