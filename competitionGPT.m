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
env.fieldSize = 50;                         % cm
env.center = [25, 25];
env.ports = [25, 0; 50, 25; 25, 50; 0, 25]; % South, East, North, West
env.portRadius = 1.5;                       % cm
env.consumeTime = 0.5;                      % s
env.leaveDistance = 4.0;                    % cm
env.itiAfterLeave = 1.0;                    % s

env.socialDist = 6.0;                       % cm
env.subordinateSlow = 0.70;
env.avoidGain = 2.0;
env.tieWindow = 0.03;                       % s

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

traitsA.noticeDelay = 0.10;
traitsA.reactionDelay = 0.00;
traitsA.vMax = 4;
traitsA.aMax = 200;
traitsA.wMax = rad(360);
traitsA.alphaMax = rad(2000);
traitsA.fov = rad(140);
traitsA.scanRateFrac = 0.60;

traitsB.noticeDelay = 0.14;
traitsB.reactionDelay = 0.02;
traitsB.vMax = 3;
traitsB.aMax = 160;
traitsB.wMax = rad(330);
traitsB.alphaMax = rad(1600);
traitsB.fov = rad(120);
traitsB.scanRateFrac = 0.60;

%% Actor-Critic params
ac.alphaV = 0.10;
ac.alphaPi = 0.05;
ac.entropy = 0.005;

%% Controller params
ctrl.kHeadingP = 8.0;
ctrl.kHeadingD = 1.5;
ctrl.kSpeed = 6.0;
ctrl.slowRadius = 8.0;
ctrl.kScanW = 10.0;

%% Realtime visualization
vis.enableRealtime = true;
vis.updateEveryNSteps = 2;
vis.tailMaxPoints = 300;
vis.drawPause = 0.001;

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
        updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage1", reward1, reward2, 0, vis.drawPause);
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
            updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage1", reward1, reward2, 0, vis.drawPause);
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
            updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage2", reward1, reward2, winner, vis.drawPause);
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
            updateRealtimeView(hRT, agent1, agent2, targetPos, tr, "Stage3", reward1, reward2, winner, vis.drawPause);
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
    agent.kSpeed = ctrl.kSpeed;
    agent.slowRadius = ctrl.slowRadius;
    agent.kScanW = ctrl.kScanW;

    agent.thetaPi = zeros(nStates, nActions);
    agent.V = zeros(nStates, 1);

    agent.pos = [0 0];
    agent.theta = 0;
    agent.v = 0;
    agent.w = 0;

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

        aCmd = agent.kSpeed * (0 - agent.v);
        aCmd = clamp(aCmd, -aMax, aMax);
    else
        if dist < 1e-9
            desiredHeading = agent.theta;
        else
            desiredHeading = atan2(vec(2), vec(1));
        end

        err = wrapToPi(desiredHeading - agent.theta);
        wDot = agent.kHeadingP * err - agent.kHeadingD * agent.w;
        wDot = clamp(wDot, -alphaMax, alphaMax);

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
    h.fig = figure('Name', 'Realtime Agent Positions', 'Color', 'w');
    h.ax = axes('Parent', h.fig);
    hold(h.ax, 'on');
    axis(h.ax, [0 env.fieldSize 0 env.fieldSize]);
    axis(h.ax, 'square');
    grid(h.ax, 'on');
    xlabel(h.ax, 'X (cm)');
    ylabel(h.ax, 'Y (cm)');

    scatter(h.ax, env.ports(:,1), env.ports(:,2), 40, [0.2 0.2 0.2], 'filled');
    h.target = plot(h.ax, NaN, NaN, 'o', 'MarkerSize', 12, 'LineWidth', 2, ...
        'MarkerFaceColor', [1.0 0.9 0.1], 'Color', [0.9 0.5 0.0]);

    h.tail1 = animatedline(h.ax, 'Color', [0.2 0.4 1.0], 'LineWidth', 1.0, ...
        'MaximumNumPoints', vis.tailMaxPoints);
    h.tail2 = animatedline(h.ax, 'Color', [1.0 0.3 0.2], 'LineWidth', 1.0, ...
        'MaximumNumPoints', vis.tailMaxPoints);

    h.body1 = plot(h.ax, NaN, NaN, 'o', 'MarkerSize', 8, ...
        'MarkerFaceColor', [0.2 0.4 1.0], 'MarkerEdgeColor', 'k');
    h.body2 = plot(h.ax, NaN, NaN, 'o', 'MarkerSize', 8, ...
        'MarkerFaceColor', [1.0 0.3 0.2], 'MarkerEdgeColor', 'k');

    h.head1 = quiver(h.ax, NaN, NaN, NaN, NaN, 0, 'Color', [0.1 0.2 0.8], 'LineWidth', 1.5);
    h.head2 = quiver(h.ax, NaN, NaN, NaN, NaN, 0, 'Color', [0.8 0.1 0.1], 'LineWidth', 1.5);

    h.info = title(h.ax, 'Realtime');
end

function updateRealtimeView(h, agent1, agent2, targetPos, tr, stageName, reward1, reward2, winner, drawPause)
    if ~isfield(h, 'fig') || ~isgraphics(h.fig)
        return;
    end

    set(h.target, 'XData', targetPos(1), 'YData', targetPos(2));

    addpoints(h.tail1, agent1.pos(1), agent1.pos(2));
    addpoints(h.tail2, agent2.pos(1), agent2.pos(2));

    set(h.body1, 'XData', agent1.pos(1), 'YData', agent1.pos(2));
    set(h.body2, 'XData', agent2.pos(1), 'YData', agent2.pos(2));

    headLen = 2.0;
    set(h.head1, 'XData', agent1.pos(1), 'YData', agent1.pos(2), ...
        'UData', headLen*cos(agent1.theta), 'VData', headLen*sin(agent1.theta));
    set(h.head2, 'XData', agent2.pos(1), 'YData', agent2.pos(2), ...
        'UData', headLen*cos(agent2.theta), 'VData', headLen*sin(agent2.theta));

    if winner == 0
        wtxt = "ongoing";
    else
        wtxt = "winner=" + string(winner);
    end

    h.info.String = sprintf('Trial %d | %s | reward: %d-%d | %s', tr, stageName, reward1, reward2, wtxt);
    drawnow limitrate;
    pause(drawPause);
end
