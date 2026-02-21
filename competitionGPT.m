%% competitionGPT.m
% Competitive agents task (2 mice, Actor-Critic)
% - 50cm x 50cm open field
% - reward ports at midpoints of 4 edges
% - one random port lights each trial
% - only first-arriving mouse gets reward
% - next trial starts 1s after winner leaves the port
% - 500 trials

clear; clc; close all;
rng(1);

%% Environment
env.fieldSize = 50;                         % [cm] フィールド一辺の長さ（正方形）
env.center = [25, 25];                      % [cm, cm] フィールド中心座標
env.ports = [25, 0; 50, 25; 25, 50; 0, 25]; % [cm] 報酬ポート座標（South, East, North, West）
env.portRadius = 1.5;                       % [cm] ポート到達判定に使う半径
env.consumeTime = 0.5;                      % [s] 勝者がポートで報酬を消費して留まる時間
env.leaveDistance = 4.0;                    % [cm] 勝者がポートを離れたとみなす距離
env.itiAfterLeave = 2.0;                    % [s] 勝者離脱後のITI（次trial開始まで）

env.socialDist = 6.0;                       % [cm] 相手がこの距離内だと社会的干渉を適用
env.subordinateSlow = 0.70;                 % [-] 劣位個体にかかる速度/角速度スケール
env.avoidGain = 2.0;                        % [-] 相手回避ベクトルの強さ
env.tieWindow = 0.03;                       % [s] 同時到達とみなす時間窓

dt = 0.02;
nTrials = 500;
decision.decisionInterval = 0.25;         % s: trial中の再選択間隔
decision.hopelessEtaGap = 1.0;            % s: これ以上ETAで不利ならignore候補
decision.hopelessDistGap = 8.0;           % cm: 距離差が十分大きい時のみignore候補

%% Action / State
% action = {IGNORE, CHASE} x {Center, S, E, N, W}
waypoints = [env.center; env.ports];
nWp = size(waypoints, 1);
nActions = 2 * nWp;

% state = lastPort(0..4) x lastWin(0/1) x advantageBin(1..3)
% advantageBin: 1=favor, 2=even, 3=unfavorable
nStates = 5 * 2 * 3;

%% Traits
rad = @(deg) deg * pi / 180;

traitsA.noticeDelay = 0.10;      % [s] 光を視野内に捉えてから「気づく」までの遅延
traitsA.reactionDelay = 0.00;    % [s] 気づいた後に運動方針へ反映されるまでの遅延
traitsA.vMax = 40;               % [cm/s] 最大並進速度
traitsA.aMax = 180;              % [cm/s^2] 最大並進加速度
traitsA.wMax = rad(360);         % [rad/s] 最大角速度
traitsA.alphaMax = rad(3600);    % [rad/s^2] 最大角加速度
traitsA.fov = rad(300)           % [rad] 視野角（頭部正面中心の扇形）
traitsA.scanRateFrac = 0.60;     % [-] 探索旋回時に使う角速度比（scan時のw目標 = wMax*この値）

traitsB.noticeDelay = 0.14;      % [s] 光に気づくまでの遅延
traitsB.reactionDelay = 0.02;    % [s] 気づいた後の行動反映遅延
traitsB.vMax = 36;               % [cm/s] 最大並進速度
traitsB.aMax = 160;              % [cm/s^2] 最大並進加速度
traitsB.wMax = rad(330);         % [rad/s] 最大角速度
traitsB.alphaMax = rad(3600);    % [rad/s^2] 最大角加速度
traitsB.fov = rad(320);          % [rad] 視野角
traitsB.scanRateFrac = 0.60;     % [-] 探索旋回時の角速度比

%% Actor-Critic params
ac.alphaV = 0.10;       % [-] criticの学習率（価値関数Vの更新幅）
ac.alphaPi = 0.05;      % [-] actorの学習率（方策パラメータの更新幅）
ac.entropy = 0.005;     % [-] エントロピー正則化の強さ（探索を維持）

%% Controller params
ctrl.kHeadingP = 5.0;   % [-] 方位誤差に対するPゲイン（高すぎると振動しやすい）
ctrl.kHeadingD = 2.2;   % [-] 角速度に対するDゲイン（減衰を強めて振動抑制）
ctrl.kHeadingI = 0.05;  % [-] 方位誤差の積分ゲイン（微小な定常偏差のみ補正）
ctrl.headingILimit = 0.12; % [rad*s] 積分状態の上限（アンチワインドアップ）
ctrl.kSpeed = 6.0;      % [-] 速度追従のPゲイン（目標速度への追従性）
ctrl.slowRadius = 8.0;  % [cm] 目標近傍で減速を開始する半径
ctrl.kScanW = 10.0;     % [-] scanモード時の角速度追従ゲイン

%% Realtime visualization
vis.enableRealtime = true;                        % [bool] リアルタイム表示のON/OFF
vis.updateEveryNSteps = 2;                        % [step] 何ステップごとに描画更新するか
vis.tailMaxPoints = 300;                          % [point] 軌跡（tail）に保持する最大点数
vis.drawPause = 0.001;                            % [s] 描画更新後の短い待機時間
vis.maxEvents = 10;                               % [count] イベントログの最大表示件数
vis.maWindow = 50;                                % [trial] 指標の移動平均窓サイズ
vis.decisionInterval = decision.decisionInterval; % [s] 再選択残り時間表示の基準

%% Build agents
agent1 = makeAgent("MouseA", 1, traitsA, ac, ctrl, nStates, nActions);
agent2 = makeAgent("MouseB", 2, traitsB, ac, ctrl, nStates, nActions);
agent1 = randomizePose(agent1, env);
agent2 = randomizePose(agent2, env);

%% Logs
log.targetPort = zeros(nTrials, 1);
log.winner = zeros(nTrials, 1);
log.trialDur = zeros(nTrials, 1);
log.state1 = zeros(nTrials, 1);
log.state2 = zeros(nTrials, 1);
log.action1 = zeros(nTrials, 1);
log.action2 = zeros(nTrials, 1);
log.cumReward1 = zeros(nTrials, 1);
log.cumReward2 = zeros(nTrials, 1);

saveTrajTrials = 10;
traj = cell(saveTrajTrials, 1);

if vis.enableRealtime
    hRT = initRealtimeView(env, vis);
else
    hRT = struct();
end

%% Main loop
tGlobal = 0;
reward1 = 0;
reward2 = 0;
lastPort = 0;
lastWin1 = 0;
lastWin2 = 0;
simStep = 0;

for tr = 1:nTrials
    targetPort = randi(4);
    targetPos = env.ports(targetPort, :);
    log.targetPort(tr) = targetPort;

    if vis.enableRealtime && isfield(hRT, 'fig') && isgraphics(hRT.fig)
        clearpoints(hRT.tail1);
        clearpoints(hRT.tail2);
        updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage1", reward1, reward2, 0, vis.decisionInterval, vis.decisionInterval, vis.drawPause);
    end

    advBin1 = advantageBin(agent1, agent2, targetPos);
    advBin2 = advantageBin(agent2, agent1, targetPos);

    s1 = encodeState(lastPort, lastWin1, advBin1);
    s2 = encodeState(lastPort, lastWin2, advBin2);

    [agent1, a1, pi1] = selectActionByContext(agent1, s1, targetPos, agent2, nWp, decision);
    [agent2, a2, pi2] = selectActionByContext(agent2, s2, targetPos, agent1, nWp, decision);

    agent1 = applyAction(agent1, a1, waypoints);
    agent2 = applyAction(agent2, a2, waypoints);
    agent1 = resetTrialState(agent1);
    agent2 = resetTrialState(agent2);

    log.state1(tr) = s1;
    log.state2(tr) = s2;
    log.action1(tr) = a1;
    log.action2(tr) = a2;

    if tr <= saveTrajTrials
        traj{tr}.p1 = [];
        traj{tr}.p2 = [];
    end

    % Stage 1: competition
    tTrialStart = tGlobal;
    winner = 0;
    dPrev1 = norm(agent1.pos - targetPos);
    dPrev2 = norm(agent2.pos - targetPos);
    elapsedSinceDecision1 = 0;
    elapsedSinceDecision2 = 0;
    sPrev1 = s1; aPrev1 = a1; piPrev1 = pi1;
    sPrev2 = s2; aPrev2 = a2; piPrev2 = pi2;

    while winner == 0
        if tr <= saveTrajTrials
            traj{tr}.p1(end+1, :) = agent1.pos;
            traj{tr}.p2(end+1, :) = agent2.pos;
        end

        agent1 = stepAgent(agent1, agent2, env, dt, 1, targetPos, waypoints);
        agent2 = stepAgent(agent2, agent1, env, dt, 1, targetPos, waypoints);
        tGlobal = tGlobal + dt;
        simStep = simStep + 1;

        if vis.enableRealtime && isfield(hRT, 'fig') && isgraphics(hRT.fig) && mod(simStep, vis.updateEveryNSteps) == 0
            updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage1", reward1, reward2, 0, decision.decisionInterval - elapsedSinceDecision1, decision.decisionInterval - elapsedSinceDecision2, vis.drawPause);
        end

        dCurr1 = norm(agent1.pos - targetPos);
        dCurr2 = norm(agent2.pos - targetPos);

        [cross1, tCross1] = crossingTime(dPrev1, dCurr1, env.portRadius, tGlobal - dt, dt);
        [cross2, tCross2] = crossingTime(dPrev2, dCurr2, env.portRadius, tGlobal - dt, dt);

        if cross1 || cross2
            winner = resolveWinner(cross1, tCross1, cross2, tCross2, agent1, agent2, env.tieWindow);
            if winner == 1
                agent1.pos = targetPos;
                agent1.v = 0;
                agent1.w = 0;
            else
                agent2.pos = targetPos;
                agent2.v = 0;
                agent2.w = 0;
            end
        end

        dPrev1 = dCurr1;
        dPrev2 = dCurr2;

        % Trial中の再選択 + TD(0)更新（中間報酬0）
        if winner == 0
            elapsedSinceDecision1 = elapsedSinceDecision1 + dt;
            elapsedSinceDecision2 = elapsedSinceDecision2 + dt;

            if elapsedSinceDecision1 >= decision.decisionInterval
                advNow1 = advantageBin(agent1, agent2, targetPos);
                sNow1 = encodeState(lastPort, lastWin1, advNow1);
                agent1 = updateActorCriticTD(agent1, sPrev1, aPrev1, 0, sNow1, piPrev1);
                [agent1, aNow1, piNow1] = selectActionByContext(agent1, sNow1, targetPos, agent2, nWp, decision);
                agent1 = applyAction(agent1, aNow1, waypoints);
                sPrev1 = sNow1; aPrev1 = aNow1; piPrev1 = piNow1;
                elapsedSinceDecision1 = 0;
            end

            if elapsedSinceDecision2 >= decision.decisionInterval
                advNow2 = advantageBin(agent2, agent1, targetPos);
                sNow2 = encodeState(lastPort, lastWin2, advNow2);
                agent2 = updateActorCriticTD(agent2, sPrev2, aPrev2, 0, sNow2, piPrev2);
                [agent2, aNow2, piNow2] = selectActionByContext(agent2, sNow2, targetPos, agent1, nWp, decision);
                agent2 = applyAction(agent2, aNow2, waypoints);
                sPrev2 = sNow2; aPrev2 = aNow2; piPrev2 = piNow2;
                elapsedSinceDecision2 = 0;
            end
        end

        if (tGlobal - tTrialStart) > 20
            winner = randi(2);
        end
    end

    log.winner(tr) = winner;
    log.trialDur(tr) = tGlobal - tTrialStart;

    r1 = double(winner == 1);
    r2 = double(winner == 2);

    reward1 = reward1 + r1;
    reward2 = reward2 + r2;
    log.cumReward1(tr) = reward1;
    log.cumReward2(tr) = reward2;

    agent1 = updateActorCriticTerminal(agent1, sPrev1, aPrev1, r1, piPrev1);
    agent2 = updateActorCriticTerminal(agent2, sPrev2, aPrev2, r2, piPrev2);

    lastPort = targetPort;
    lastWin1 = r1;
    lastWin2 = r2;

    % Stage 2: consumption + leave
    agent1.consumeTimer = env.consumeTime * r1;
    agent2.consumeTimer = env.consumeTime * r2;
    agent1.leftPort = ~logical(r1);
    agent2.leftPort = ~logical(r2);

    leavingDone = false;
    tLeaveStart = tGlobal;
    while ~leavingDone
        agent1 = stepAgent(agent1, agent2, env, dt, 2, targetPos, waypoints);
        agent2 = stepAgent(agent2, agent1, env, dt, 2, targetPos, waypoints);
        tGlobal = tGlobal + dt;
        simStep = simStep + 1;

        if vis.enableRealtime && isfield(hRT, 'fig') && isgraphics(hRT.fig) && mod(simStep, vis.updateEveryNSteps) == 0
            updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage2", reward1, reward2, winner, 0, 0, vis.drawPause);
        end

        if winner == 1
            leavingDone = agent1.leftPort;
        else
            leavingDone = agent2.leftPort;
        end

        if (tGlobal - tLeaveStart) > 10
            leavingDone = true;
        end
    end

    % Stage 3: ITI
    iti = env.itiAfterLeave;
    while iti > 0
        agent1 = stepAgent(agent1, agent2, env, dt, 3, targetPos, waypoints);
        agent2 = stepAgent(agent2, agent1, env, dt, 3, targetPos, waypoints);
        tGlobal = tGlobal + dt;
        iti = iti - dt;
        simStep = simStep + 1;

        if vis.enableRealtime && isfield(hRT, 'fig') && isgraphics(hRT.fig) && mod(simStep, vis.updateEveryNSteps) == 0
            updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage3", reward1, reward2, winner, 0, 0, vis.drawPause);
        end
    end
end

%% Result
fprintf('=== Result (%d trials) ===\n', nTrials);
fprintf('%s wins: %d\n', agent1.name, reward1);
fprintf('%s wins: %d\n', agent2.name, reward2);

figure('Name', 'Cumulative Reward');
plot(log.cumReward1, 'LineWidth', 1.4); hold on;
plot(log.cumReward2, 'LineWidth', 1.4);
xlabel('Trial'); ylabel('Cumulative Reward');
legend(agent1.name, agent2.name, 'Location', 'northwest');
grid on;

labels = actionLabels();
figure('Name', 'Policy Preference (final)');
subplot(2,1,1);
bar(mean(agent1.thetaPi, 1));
title(agent1.name + " actor preference");
xticks(1:nActions); xticklabels(labels); xtickangle(40); grid on;
subplot(2,1,2);
bar(mean(agent2.thetaPi, 1));
title(agent2.name + " actor preference");
xticks(1:nActions); xticklabels(labels); xtickangle(40); grid on;

if saveTrajTrials > 0
    figure('Name', 'Trajectories (first trials)');
    nshow = min(saveTrajTrials, nTrials);
    for i = 1:nshow
        subplot(ceil(nshow/2), 2, i);
        plot(traj{i}.p1(:,1), traj{i}.p1(:,2), '-'); hold on;
        plot(traj{i}.p2(:,1), traj{i}.p2(:,2), '-');
        scatter(env.ports(:,1), env.ports(:,2), 30, 'filled');
        xlim([0 env.fieldSize]); ylim([0 env.fieldSize]); axis square;
        title(sprintf('Trial %d / port %d / winner %d', i, log.targetPort(i), log.winner(i)));
        grid on;
    end
    legend(agent1.name, agent2.name, 'Ports');
end

%% Local functions
function agent = makeAgent(name, dominanceRank, traits, ac, ctrl, nStates, nActions)
    agent.name = char(name);
    agent.dominance = dominanceRank;

    agent.noticeDelay = traits.noticeDelay;
    agent.reactionDelay = traits.reactionDelay;
    agent.vMax = traits.vMax;
    agent.aMax = traits.aMax;
    agent.wMax = traits.wMax;
    agent.alphaMax = traits.alphaMax;
    agent.fov = traits.fov;
    agent.scanRateFrac = traits.scanRateFrac;

    agent.alphaV = ac.alphaV;
    agent.alphaPi = ac.alphaPi;
    agent.entropy = ac.entropy;

    agent.kHeadingP = ctrl.kHeadingP;
    agent.kHeadingD = ctrl.kHeadingD;
    agent.kHeadingI = ctrl.kHeadingI;
    agent.headingILimit = ctrl.headingILimit;
    agent.kSpeed = ctrl.kSpeed;
    agent.slowRadius = ctrl.slowRadius;
    agent.kScanW = ctrl.kScanW;

    agent.thetaPi = zeros(nStates, nActions);
    agent.V = zeros(nStates, 1);

    agent.pos = [0 0];
    agent.theta = 0;
    agent.v = 0;
    agent.w = 0;
    agent.headingErrInt = 0;

    agent.wpIdx = 1;
    agent.chaseLight = 1;
    agent.waypoint = [0 0];

    agent.detected = false;
    agent.visibleTime = 0;
    agent.reactionTimer = 0;
    agent.scanDir = 1;
    agent.searchPortIdx = 1;

    agent.consumeTimer = 0;
    agent.leftPort = true;
end

function agent = randomizePose(agent, env)
    agent.pos = rand(1,2) * env.fieldSize;
    agent.theta = 2*pi*rand;
    agent.v = 0;
    agent.w = 0;
    agent.headingErrInt = 0;
end

function s = encodeState(lastPort, didWin, advBin)
    s = ((lastPort * 2 + didWin) * 3) + advBin;
end

function [agent, action, pi] = selectActionSoftmax(agent, s)
    z = agent.thetaPi(s, :);
    z = z - max(z);
    expz = exp(z);
    pi = expz / sum(expz);

    r = rand;
    c = cumsum(pi);
    action = find(r <= c, 1, 'first');
    if isempty(action)
        action = numel(pi);
    end
end

function [agent, action, piFull] = selectActionByContext(agent, s, targetPos, other, nWp, decision)
    if shouldIgnoreNow(agent, other, targetPos, decision)
        allowed = 1:nWp;            % IGNORE-* のみ
    else
        allowed = (nWp+1):(2*nWp);  % CHASE-* のみ（基本）
    end
    [action, piFull] = sampleRestrictedSoftmax(agent.thetaPi(s, :), allowed);
end

function [action, piFull] = sampleRestrictedSoftmax(logits, allowed)
    nA = numel(logits);
    z = logits(:)';
    zAllowed = z(allowed);
    zAllowed = zAllowed - max(zAllowed);
    pAllowed = exp(zAllowed);
    pAllowed = pAllowed / sum(pAllowed);

    c = cumsum(pAllowed);
    r = rand;
    k = find(r <= c, 1, 'first');
    if isempty(k)
        k = numel(allowed);
    end
    action = allowed(k);

    piFull = zeros(1, nA);
    piFull(allowed) = pAllowed;
end

function tf = shouldIgnoreNow(agent, other, targetPos, decision)
    etaSelf = estimateETA(agent, targetPos);
    etaOther = estimateETA(other, targetPos);
    dGap = norm(agent.pos - targetPos) - norm(other.pos - targetPos);
    tf = (etaSelf - etaOther >= decision.hopelessEtaGap) && (dGap >= decision.hopelessDistGap);
end

function agent = updateActorCriticTD(agent, s, a, r, sNext, pi)
    delta = r + agent.V(sNext) - agent.V(s);
    agent.V(s) = agent.V(s) + agent.alphaV * delta;

    grad = -pi;
    grad(a) = grad(a) + 1;
    entGrad = -(log(max(pi, 1e-12)) + 1);
    agent.thetaPi(s, :) = agent.thetaPi(s, :) + agent.alphaPi * (delta * grad + agent.entropy * entGrad);
end

function agent = updateActorCriticTerminal(agent, s, a, r, pi)
    delta = r - agent.V(s);
    agent.V(s) = agent.V(s) + agent.alphaV * delta;

    grad = -pi;
    grad(a) = grad(a) + 1;
    entGrad = -(log(max(pi, 1e-12)) + 1);
    agent.thetaPi(s, :) = agent.thetaPi(s, :) + agent.alphaPi * (delta * grad + agent.entropy * entGrad);
end

function agent = applyAction(agent, action, waypoints)
    nWp = size(waypoints, 1);
    chase = floor((action - 1) / nWp);
    wpIdx = mod(action - 1, nWp) + 1;

    agent.chaseLight = chase;
    agent.wpIdx = wpIdx;
    agent.waypoint = waypoints(wpIdx, :);
end

function agent = resetTrialState(agent)
    agent.detected = false;
    agent.visibleTime = 0;
    agent.reactionTimer = 0;
    d = sign(randn);
    if d == 0
        d = 1;
    end
    agent.scanDir = d;
    agent.headingErrInt = 0;
end

function b = advantageBin(agent, other, targetPos)
    etaSelf = estimateETA(agent, targetPos);
    etaOther = estimateETA(other, targetPos);
    adv = etaSelf - etaOther;

    if adv < -0.4
        b = 1;
    elseif adv > 0.4
        b = 3;
    else
        b = 2;
    end
end

function eta = estimateETA(agent, targetPos)
    d = norm(agent.pos - targetPos);
    head = atan2(targetPos(2) - agent.pos(2), targetPos(1) - agent.pos(1));
    err = abs(wrapToPi(head - agent.theta));
    turnTime = err / max(agent.wMax, 1e-6);
    moveTime = d / max(agent.vMax, 1e-6);
    visible = (err <= agent.fov / 2);
    detectCost = (~visible) * (agent.noticeDelay + 0.2);
    eta = turnTime + detectCost + agent.reactionDelay + moveTime;
end

function winner = resolveWinner(c1, t1, c2, t2, a1, a2, tieWindow)
    if c1 && ~c2
        winner = 1;
        return;
    end
    if ~c1 && c2
        winner = 2;
        return;
    end

    if abs(t1 - t2) <= tieWindow
        if a1.dominance < a2.dominance
            winner = 1;
        elseif a2.dominance < a1.dominance
            winner = 2;
        else
            winner = randi(2);
        end
    else
        winner = 1 + (t2 < t1);
    end
end

function agent = stepAgent(agent, other, env, dt, stage, targetPos, waypoints)
    effV = agent.vMax;
    effA = agent.aMax;
    effW = agent.wMax;
    effAlpha = agent.alphaMax;

    dSoc = norm(agent.pos - other.pos);
    sub = agent.dominance > other.dominance;
    if sub && dSoc < env.socialDist
        effV = effV * env.subordinateSlow;
        effA = effA * env.subordinateSlow;
        effW = effW * env.subordinateSlow;
        effAlpha = effAlpha * env.subordinateSlow;
    end

    goalPos = agent.waypoint;
    vGoal = effV;
    scanMode = false;

    if stage == 1
        if ~agent.detected
            if isTargetVisible(agent, targetPos)
                agent.visibleTime = agent.visibleTime + dt;
                if agent.visibleTime >= agent.noticeDelay
                    agent.detected = true;
                    agent.reactionTimer = agent.reactionDelay;
                end
            else
                agent.visibleTime = 0;
            end
        else
            agent.reactionTimer = max(0, agent.reactionTimer - dt);
        end

        if agent.chaseLight == 1
            if agent.detected && agent.reactionTimer <= 0
                goalPos = targetPos;
            else
                if norm(agent.pos - agent.waypoint) < 1.0
                    goalPos = agent.pos;
                    vGoal = 0;
                    scanMode = true;
                else
                    goalPos = agent.waypoint;
                end
            end
        else
            % IGNORE時は選択アクションの目的地へ向かう（巡回固定しない）
            goalPos = agent.waypoint;
            if norm(agent.pos - goalPos) < 1.0
                vGoal = 0;
                scanMode = true;
            end
        end

    elseif stage == 2
        if agent.consumeTimer > 0
            goalPos = targetPos;
            vGoal = 0;
            agent.consumeTimer = agent.consumeTimer - dt;
        else
            if ~agent.leftPort
                goalPos = env.center;
                if norm(agent.pos - targetPos) > env.leaveDistance
                    agent.leftPort = true;
                end
            end
        end

    else
        goalPos = agent.waypoint;
    end

    if sub && dSoc < env.socialDist
        avoidDir = (agent.pos - other.pos) / max(dSoc, 1e-6);
        goalPos = goalPos + env.avoidGain * (env.socialDist - dSoc) * avoidDir;
    end

    agent = moveUnicycle(agent, goalPos, vGoal, effV, effA, effW, effAlpha, env, dt, scanMode);
end

function agent = moveUnicycle(agent, goalPos, vGoal, vMax, aMax, wMax, alphaMax, env, dt, scanMode)
    vec = goalPos - agent.pos;
    dist = norm(vec);

    if dist < agent.slowRadius
        vGoal = vGoal * (dist / max(agent.slowRadius, 1e-6));
    end

    if scanMode
        desiredW = agent.scanDir * (agent.scanRateFrac * wMax);
        wDot = agent.kScanW * (desiredW - agent.w);
        wDot = clamp(wDot, -alphaMax, alphaMax);
        agent.headingErrInt = 0;

        aCmd = agent.kSpeed * (0 - agent.v);
        aCmd = clamp(aCmd, -aMax, aMax);
    else
        if dist < 1e-9
            desiredHeading = agent.theta;
        else
            desiredHeading = atan2(vec(2), vec(1));
        end

        err = wrapToPi(desiredHeading - agent.theta);

        % PID integral term with leak + anti-windup to reduce heading oscillation.
        if abs(err) < (pi/3)
            agent.headingErrInt = 0.98 * agent.headingErrInt + err * dt;
        else
            % Far from target heading: bleed integral to avoid overshoot.
            agent.headingErrInt = 0.90 * agent.headingErrInt;
        end
        if abs(err) < 0.05
            agent.headingErrInt = 0;
        end
        agent.headingErrInt = clamp(agent.headingErrInt, -agent.headingILimit, agent.headingILimit);

        % Use saturated proportional term to avoid overly aggressive response.
        pTerm = agent.kHeadingP * tanh(2.0 * err);
        wDotUnsat = pTerm + agent.kHeadingI * agent.headingErrInt - agent.kHeadingD * agent.w;
        wDot = clamp(wDotUnsat, -alphaMax, alphaMax);

        % Back-calculation anti-windup when angular acceleration saturates.
        if abs(agent.kHeadingI) > 1e-9
            aw = (wDotUnsat - wDot) / agent.kHeadingI;
            agent.headingErrInt = clamp(agent.headingErrInt - 0.5 * aw * dt, -agent.headingILimit, agent.headingILimit);
        end

        % Deadband near target heading: damp out residual oscillation.
        if abs(err) < 0.03
            agent.headingErrInt = 0;
            wDot = clamp(-2.0 * agent.w, -alphaMax, alphaMax);
        end

        % If heading is far from goal direction, rotate first then accelerate.
        if abs(err) > pi/2
            vGoal = 0;
        else
            vGoal = vGoal * max(0, cos(err))^1.5;
        end

        aCmd = agent.kSpeed * (vGoal - agent.v);
        aCmd = clamp(aCmd, -aMax, aMax);
    end

    agent.w = clamp(agent.w + wDot * dt, -wMax, wMax);
    agent.theta = wrapToPi(agent.theta + agent.w * dt);

    agent.v = clamp(agent.v + aCmd * dt, 0, vMax);
    agent.pos = agent.pos + agent.v * [cos(agent.theta), sin(agent.theta)] * dt;

    [agent.pos, agent.theta] = reflectAtWalls(agent.pos, agent.theta, env.fieldSize);
end

function visible = isTargetVisible(agent, targetPos)
    vec = targetPos - agent.pos;
    ang = atan2(vec(2), vec(1));
    err = wrapToPi(ang - agent.theta);
    visible = abs(err) <= agent.fov / 2;
end

function [crossed, tCross] = crossingTime(dPrev, dCurr, r, t0, dt)
    crossed = false;
    tCross = inf;

    if dPrev <= r
        crossed = true;
        tCross = t0;
        return;
    end

    if dPrev > r && dCurr <= r
        crossed = true;
        frac = (dPrev - r) / max(dPrev - dCurr, 1e-12);
        frac = clamp(frac, 0, 1);
        tCross = t0 + frac * dt;
    end
end

function [pos, theta] = reflectAtWalls(pos, theta, L)
    if pos(1) < 0
        pos(1) = 0;
        theta = wrapToPi(pi - theta);
    elseif pos(1) > L
        pos(1) = L;
        theta = wrapToPi(pi - theta);
    end

    if pos(2) < 0
        pos(2) = 0;
        theta = wrapToPi(-theta);
    elseif pos(2) > L
        pos(2) = L;
        theta = wrapToPi(-theta);
    end
end

function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end

function a = wrapToPi(a)
    a = mod(a + pi, 2*pi) - pi;
end

function labels = actionLabels()
    wp = ["Center", "South", "East", "North", "West"];
    labels = strings(1, 10);
    k = 1;
    for mode = ["IGNORE", "CHASE"]
        for i = 1:5
            labels(k) = mode + "-" + wp(i);
            k = k + 1;
        end
    end
end

function h = initRealtimeView(env, vis)
    h.fig = figure('Name', 'Realtime Agent Dashboard', 'Color', 'w');
    tl = tiledlayout(h.fig, 4, 4, 'Padding', 'compact', 'TileSpacing', 'compact');

    h.axMain = nexttile(tl, [3 2]);
    hold(h.axMain, 'on');
    axis(h.axMain, [0 env.fieldSize 0 env.fieldSize]);
    axis(h.axMain, 'square');
    grid(h.axMain, 'on');
    xlabel(h.axMain, 'X (cm)');
    ylabel(h.axMain, 'Y (cm)');
    scatter(h.axMain, env.ports(:,1), env.ports(:,2), 36, [0.2 0.2 0.2], 'filled');
    h.target = plot(h.axMain, NaN, NaN, 'o', 'MarkerSize', 12, 'LineWidth', 2, ...
        'MarkerFaceColor', [1.0 0.9 0.1], 'Color', [0.9 0.5 0.0]);
    h.tail1 = animatedline(h.axMain, 'Color', [0.2 0.4 1.0], 'LineWidth', 1.0, ...
        'MaximumNumPoints', vis.tailMaxPoints);
    h.tail2 = animatedline(h.axMain, 'Color', [1.0 0.3 0.2], 'LineWidth', 1.0, ...
        'MaximumNumPoints', vis.tailMaxPoints);
    h.body1 = plot(h.axMain, NaN, NaN, 'o', 'MarkerSize', 8, ...
        'MarkerFaceColor', [0.2 0.4 1.0], 'MarkerEdgeColor', 'k');
    h.body2 = plot(h.axMain, NaN, NaN, 'o', 'MarkerSize', 8, ...
        'MarkerFaceColor', [1.0 0.3 0.2], 'MarkerEdgeColor', 'k');
    h.head1 = quiver(h.axMain, NaN, NaN, NaN, NaN, 0, 'Color', [0.1 0.2 0.8], 'LineWidth', 1.3);
    h.head2 = quiver(h.axMain, NaN, NaN, NaN, NaN, 0, 'Color', [0.8 0.1 0.1], 'LineWidth', 1.3);
    h.goal1 = quiver(h.axMain, NaN, NaN, NaN, NaN, 0, 'Color', [0.3 0.7 1.0], 'LineStyle', '--', 'LineWidth', 1.0);
    h.goal2 = quiver(h.axMain, NaN, NaN, NaN, NaN, 0, 'Color', [1.0 0.5 0.4], 'LineStyle', '--', 'LineWidth', 1.0);
    h.fov1 = patch(h.axMain, NaN, NaN, [0.2 0.4 1.0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
    h.fov2 = patch(h.axMain, NaN, NaN, [1.0 0.3 0.2], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
    h.mainTitle = title(h.axMain, 'Realtime');

    labels = actionLabels();
    nActions = numel(labels);
    h.axPolicy1 = nexttile(tl);
    h.policyBar1 = bar(h.axPolicy1, zeros(1, nActions), 'FaceColor', [0.2 0.4 1.0]);
    ylim(h.axPolicy1, [0 1]); grid(h.axPolicy1, 'on'); title(h.axPolicy1, 'Policy A1');
    xticks(h.axPolicy1, 1:nActions); xticklabels(h.axPolicy1, labels); xtickangle(h.axPolicy1, 45);

    h.axPolicy2 = nexttile(tl);
    h.policyBar2 = bar(h.axPolicy2, zeros(1, nActions), 'FaceColor', [1.0 0.3 0.2]);
    ylim(h.axPolicy2, [0 1]); grid(h.axPolicy2, 'on'); title(h.axPolicy2, 'Policy A2');
    xticks(h.axPolicy2, 1:nActions); xticklabels(h.axPolicy2, labels); xtickangle(h.axPolicy2, 45);

    h.axComp = nexttile(tl);
    h.compBar = bar(h.axComp, [0 0], 'FaceColor', 'flat');
    h.compBar.CData = [0.2 0.4 1.0; 1.0 0.3 0.2];
    ylim(h.axComp, [0 env.fieldSize]); grid(h.axComp, 'on');
    title(h.axComp, 'Distance To Target');
    xticks(h.axComp, [1 2]); xticklabels(h.axComp, {'A1','A2'});
    h.compText = text(h.axComp, 1.5, env.fieldSize*0.9, '', 'HorizontalAlignment', 'center');

    h.axSpeed = nexttile(tl);
    h.speedBar = bar(h.axSpeed, zeros(1,4), 'FaceColor', 'flat');
    h.speedBar.CData = [0.2 0.4 1.0; 0.2 0.7 1.0; 1.0 0.3 0.2; 1.0 0.6 0.3];
    ylim(h.axSpeed, [0 1]); grid(h.axSpeed, 'on'); title(h.axSpeed, 'Speed Meters');
    xticks(h.axSpeed, 1:4); xticklabels(h.axSpeed, {'A1 v','A1 w','A2 v','A2 w'});

    h.axSensor = nexttile(tl);
    h.sensorBar = bar(h.axSensor, zeros(1,6), 'FaceColor', 'flat');
    h.sensorBar.CData = [0.2 0.4 1.0; 0.2 0.7 1.0; 0.4 0.8 1.0; 1.0 0.3 0.2; 1.0 0.6 0.3; 1.0 0.8 0.4];
    ylim(h.axSensor, [0 1]); grid(h.axSensor, 'on'); title(h.axSensor, 'Sensor');
    xticks(h.axSensor, 1:6); xticklabels(h.axSensor, {'A1 vis','A1 det','A1 notice','A2 vis','A2 det','A2 notice'});

    h.axTrend = nexttile(tl);
    hold(h.axTrend, 'on');
    h.trWin1 = plot(h.axTrend, NaN, NaN, 'Color', [0.2 0.4 1.0], 'LineWidth', 1.2);
    h.trWin2 = plot(h.axTrend, NaN, NaN, 'Color', [1.0 0.3 0.2], 'LineWidth', 1.2);
    h.trIgnore1 = plot(h.axTrend, NaN, NaN, '--', 'Color', [0.2 0.4 1.0], 'LineWidth', 1.0);
    h.trIgnore2 = plot(h.axTrend, NaN, NaN, '--', 'Color', [1.0 0.3 0.2], 'LineWidth', 1.0);
    h.trDur = plot(h.axTrend, NaN, NaN, 'k:', 'LineWidth', 1.2);
    ylim(h.axTrend, [0 1]); grid(h.axTrend, 'on'); title(h.axTrend, 'Running Metrics');
    legend(h.axTrend, {'A1 winMA','A2 winMA','A1 ignore','A2 ignore','meanDur/20'}, 'Location', 'southoutside');

    h.axEvents = nexttile(tl);
    axis(h.axEvents, 'off');
    h.eventText = text(h.axEvents, 0, 1, '', 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontName', 'Consolas', 'FontSize', 9);

    h.env = env;
    h.vis = vis;
end

function updateRealtimeView(h, agent1, agent2, targetPos, tr, stageName, reward1, reward2, winner, rem1, rem2, drawPause)
    if ~isfield(h, 'fig') || ~isgraphics(h.fig)
        return;
    end

    persistent events prevAction1 prevAction2 prevDetected1 prevDetected2 ...
        winMA1 winMA2 ignoreRate1 ignoreRate2 meanDur trialDurBuf ignoreCount1 ignoreCount2 decisionCount1
    if isempty(events)
        events = {};
        prevAction1 = -1; prevAction2 = -1;
        prevDetected1 = false; prevDetected2 = false;
        winMA1 = zeros(1, tr); winMA2 = zeros(1, tr); ignoreRate1 = zeros(1, tr); ignoreRate2 = zeros(1, tr); meanDur = zeros(1, tr);
        trialDurBuf = [];
        ignoreCount1 = 0; ignoreCount2 = 0; decisionCount1 = 0;
    end
    if numel(winMA1) < tr
        winMA1(tr) = winMA1(end); winMA2(tr) = winMA2(end);
        ignoreRate1(tr) = ignoreRate1(end); ignoreRate2(tr) = ignoreRate2(end);
        meanDur(tr) = meanDur(end);
    end

    labels = actionLabels();
    nWp = 5;
    a1 = agent1.wpIdx + nWp * agent1.chaseLight;
    a2 = agent2.wpIdx + nWp * agent2.chaseLight;
    if a1 ~= prevAction1
        decisionCount1 = decisionCount1 + 1;
        ignoreCount1 = ignoreCount1 + double(a1 <= nWp);
        events = appendEvent(events, sprintf('Trial %d A1 action=%s', tr, char(labels(a1))), h.vis.maxEvents);
        prevAction1 = a1;
    end
    if a2 ~= prevAction2
        ignoreCount2 = ignoreCount2 + double(a2 <= nWp);
        events = appendEvent(events, sprintf('Trial %d A2 action=%s', tr, char(labels(a2))), h.vis.maxEvents);
        prevAction2 = a2;
    end
    if (~prevDetected1) && agent1.detected
        events = appendEvent(events, sprintf('Trial %d A1 detected', tr), h.vis.maxEvents);
    end
    if (~prevDetected2) && agent2.detected
        events = appendEvent(events, sprintf('Trial %d A2 detected', tr), h.vis.maxEvents);
    end
    prevDetected1 = agent1.detected;
    prevDetected2 = agent2.detected;
    if winner > 0
        events = appendEvent(events, sprintf('Trial %d winner=A%d', tr, winner), h.vis.maxEvents);
    end

    if strcmp(stageName, "Stage3") && winner > 0
        trialDurBuf(end+1) = max(rem1, rem2);
        if numel(trialDurBuf) > h.vis.maWindow
            trialDurBuf = trialDurBuf(end-h.vis.maWindow+1:end);
        end
        winMA1(tr) = mean(reward1 >= max(0, reward1-h.vis.maWindow+1)); %#ok<*NBRAK>
        winMA2(tr) = mean(reward2 >= max(0, reward2-h.vis.maWindow+1));
        ignoreRate1(tr) = ignoreCount1 / max(decisionCount1, 1);
        ignoreRate2(tr) = ignoreCount2 / max(decisionCount1, 1);
        meanDur(tr) = mean(trialDurBuf);
    end

    set(h.target, 'XData', targetPos(1), 'YData', targetPos(2));
    addpoints(h.tail1, agent1.pos(1), agent1.pos(2));
    addpoints(h.tail2, agent2.pos(1), agent2.pos(2));
    set(h.body1, 'XData', agent1.pos(1), 'YData', agent1.pos(2));
    set(h.body2, 'XData', agent2.pos(1), 'YData', agent2.pos(2));

    headLen = 2.0;
    set(h.head1, 'XData', agent1.pos(1), 'YData', agent1.pos(2), 'UData', headLen*cos(agent1.theta), 'VData', headLen*sin(agent1.theta));
    set(h.head2, 'XData', agent2.pos(1), 'YData', agent2.pos(2), 'UData', headLen*cos(agent2.theta), 'VData', headLen*sin(agent2.theta));

    goal1 = inferGoal(agent1, targetPos, h.env, stageName);
    goal2 = inferGoal(agent2, targetPos, h.env, stageName);
    v1 = goal1 - agent1.pos; v2 = goal2 - agent2.pos;
    set(h.goal1, 'XData', agent1.pos(1), 'YData', agent1.pos(2), 'UData', v1(1), 'VData', v1(2));
    set(h.goal2, 'XData', agent2.pos(1), 'YData', agent2.pos(2), 'UData', v2(1), 'VData', v2(2));

    [x1, y1] = fovSector(agent1.pos, agent1.theta, agent1.fov, 8);
    [x2, y2] = fovSector(agent2.pos, agent2.theta, agent2.fov, 8);
    set(h.fov1, 'XData', x1, 'YData', y1);
    set(h.fov2, 'XData', x2, 'YData', y2);

    adv1 = advantageBin(agent1, agent2, targetPos);
    adv2 = advantageBin(agent2, agent1, targetPos);
    s1 = encodeState(0, 0, adv1);
    s2 = encodeState(0, 0, adv2);
    pi1 = softmaxVec(agent1.thetaPi(s1,:));
    pi2 = softmaxVec(agent2.thetaPi(s2,:));
    set(h.policyBar1, 'YData', pi1);
    set(h.policyBar2, 'YData', pi2);

    d1 = norm(agent1.pos - targetPos); d2 = norm(agent2.pos - targetPos);
    set(h.compBar, 'YData', [d1 d2]);
    set(h.compText, 'String', sprintf('dA1-dA2=%+.2f', d1-d2));

    set(h.speedBar, 'YData', [agent1.v/max(agent1.vMax,1e-6), abs(agent1.w)/max(agent1.wMax,1e-6), ...
                              agent2.v/max(agent2.vMax,1e-6), abs(agent2.w)/max(agent2.wMax,1e-6)]);
    vis1 = double(isTargetVisible(agent1, targetPos));
    vis2 = double(isTargetVisible(agent2, targetPos));
    set(h.sensorBar, 'YData', [vis1, double(agent1.detected), clamp(agent1.visibleTime/max(agent1.noticeDelay,1e-6),0,1), ...
                               vis2, double(agent2.detected), clamp(agent2.visibleTime/max(agent2.noticeDelay,1e-6),0,1)]);

    idx = 1:max(1,tr);
    set(h.trWin1, 'XData', idx, 'YData', padTo(winMA1, tr));
    set(h.trWin2, 'XData', idx, 'YData', padTo(winMA2, tr));
    set(h.trIgnore1, 'XData', idx, 'YData', padTo(ignoreRate1, tr));
    set(h.trIgnore2, 'XData', idx, 'YData', padTo(ignoreRate2, tr));
    set(h.trDur, 'XData', idx, 'YData', min(1, padTo(meanDur, tr)/20));
    xlim(h.axTrend, [1 max(2,tr)]);

    set(h.axMain, 'Color', stageColor(stageName));
    eta1 = estimateETA(agent1, targetPos);
    eta2 = estimateETA(agent2, targetPos);
    h.mainTitle.String = sprintf('Trial %d | %s | reward %d-%d | action[%s,%s] | adv[%d,%d] | rem[%.2f,%.2f]s | ETA[%.2f,%.2f]', ...
        tr, char(stageName), reward1, reward2, char(labels(a1)), char(labels(a2)), adv1, adv2, max(0,rem1), max(0,rem2), eta1, eta2);

    if isempty(events), events = {'(no events)'}; end
    summary = sprintf('status=%s\ndist=[%.2f, %.2f]\nETA=[%.2f, %.2f]\n--- events ---\n%s', ...
        ternary(winner>0, sprintf('winner=A%d',winner), 'ongoing'), d1, d2, eta1, eta2, strjoin(events, newline));
    set(h.eventText, 'String', summary);

    drawnow limitrate;
    pause(drawPause);
end

function y = padTo(x, n)
    if isempty(x), y = zeros(1,n); return; end
    y = x(1:min(end,n));
    if numel(y) < n
        y(end+1:n) = y(end);
    end
end

function p = softmaxVec(z)
    z = z - max(z);
    ez = exp(z);
    p = ez / sum(ez);
end

function c = stageColor(stageName)
    if strcmp(stageName, "Stage1")
        c = [1.00 1.00 1.00];
    elseif strcmp(stageName, "Stage2")
        c = [0.96 1.00 0.96];
    else
        c = [0.96 0.97 1.00];
    end
end

function [x, y] = fovSector(pos, theta, fov, radius)
    ang = linspace(theta - fov/2, theta + fov/2, 24);
    x = [pos(1), pos(1) + radius*cos(ang), pos(1)];
    y = [pos(2), pos(2) + radius*sin(ang), pos(2)];
end

function g = inferGoal(agent, targetPos, env, stageName)
    if strcmp(stageName, "Stage1")
        if agent.chaseLight == 1 && agent.detected && agent.reactionTimer <= 0
            g = targetPos;
        else
            g = agent.waypoint;
        end
    elseif strcmp(stageName, "Stage2")
        if agent.consumeTimer > 0
            g = targetPos;
        elseif ~agent.leftPort
            g = env.center;
        else
            g = agent.waypoint;
        end
    else
        g = agent.waypoint;
    end
end

function events = appendEvent(events, msg, maxEvents)
    events{end+1} = msg;
    if numel(events) > maxEvents
        events = events(end-maxEvents+1:end);
    end
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
