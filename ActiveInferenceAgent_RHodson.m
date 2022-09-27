clear
%%% Hyper Params %%%
% true_food_source_1 = 13;
% true_food_source_2 = 91;
% true_food_source_3 = 92;
% true_water_source_1 = 28;
% true_water_source_2 = 93;
% true_sleep_source_1 = 15;
% true_sleep_source_2 = 100;
global remembered_count
previous_positions = [];
[true_food_source_1, previous_positions] = randPos(previous_positions);
[true_food_source_2, previous_positions] = randPos(previous_positions);
[true_food_source_3, previous_positions] = randPos(previous_positions);
[true_water_source_1, previous_positions] = randPos(previous_positions);
[true_water_source_2, previous_positions] = randPos(previous_positions);
[true_sleep_source_1, previous_positions] = randPos(previous_positions);
[true_sleep_source_2, previous_positions] = randPos(previous_positions);
global hill_1
global deterministic_nodes
global stochastic_nodes
%[hill, previous_positions] = randPos(previous_positions);
hill_1 = 13;
global unviable_action_count;
hills = [hill_1];
global hill_1_visited;
hill_1_visited = 0;
global memory_count;
% hill = 41;
true_food_source_1 = 7;
true_food_source_2 = 9;
true_food_source_3 =19;
true_food_source_4 = 17;
% true_water_source_1 = 22;
% true_water_source_2 = 28;
% true_water_source_3 = 79;
% true_water_source_4 = 73;
% true_sleep_source_1 = 23;
% true_sleep_source_2 = 29;
% true_sleep_source_3 = 89;
% true_sleep_source_4 = 82;
 food_locations = [true_food_source_1, true_food_source_2, true_food_source_3] ;
% water_locations = [true_water_source_1, true_water_source_2];
% sleep_locations = [true_sleep_source_1, true_sleep_source_2];
resource_locations = [food_locations];
swamp_active = 0;
food_cutoff = 11;
water_cutoff = 8;
sleep_cutoff = 6;
traj_count(1) = 0;

num_states = 25;
num_states_low = 25;
global Q_table;
global agent_model;
Q_table = zeros(5,5,num_states_low);
A{1}(:,:,:) = zeros(num_states,num_states,4);
a{1}(:,:,:) = zeros(num_states,num_states,4);
for i = 1:num_states
    A{1}(i,i,:) = 1;
    a{1}(i,i,:) = 1;
end
%a{1} = a{1}+1;
%a{1} = a{1}*5;
A{2}(:,:,:) = zeros(2,num_states,4);
a{2}(:,:,:) = ones(2,num_states,4);
%a{2} = a{2}/numel(a{2}(1,1,:));
A{2}(1,:,:) = 1;% empty area cell
A{2}(2,true_food_source_1,1) = 1;
A{2}(1,true_food_source_1,1) = 0;
A{2}(2,true_food_source_2,2) = 1;
A{2}(1,true_food_source_2,2) = 0;
A{2}(2,true_food_source_3,3) = 1;
A{2}(1,true_food_source_3,3) = 0;
A{2}(2,true_food_source_4,4) = 1;
A{2}(1,true_food_source_4,4) = 0;
% A{2}(3,true_water_source_1,1) = 1;
% A{2}(1,true_water_source_1,1) = 0;
% A{2}(3,true_water_source_2,2) = 1;
% A{2}(1,true_water_source_2,2) = 0;
% A{2}(3,true_water_source_3,3) = 1;
% A{2}(1,true_water_source_3,3) = 0;
% A{2}(3,true_water_source_4,4) = 1;
% A{2}(1,true_water_source_4,4) = 0;
% A{2}(4,true_sleep_source_1,1) = 1;
% A{2}(1,true_sleep_source_1,1) = 0;
% A{2}(4,true_sleep_source_2,2) = 1;
% A{2}(1,true_sleep_source_2,2) = 0;
% A{2}(4,true_sleep_source_3,3) = 1;
% A{2}(1,true_sleep_source_3,3) = 0;
% A{2}(4,true_sleep_source_4,4) = 1;
% A{2}(1,true_sleep_source_4,4) = 0;
A{3}(:,:,:) = zeros(5,num_states,4);
%A{3} = A{3}/numel(A{3}(:,1,1));
A{3}(5,:,:) = 1;
A{3}(1,hill_1,1) = 1;
A{3}(5,hill_1,1) = 0;
A{3}(2,hill_1,2) = 1;
A{3}(5,hill_1,2) = 0;
A{3}(3,hill_1,3) = 1;
A{3}(5,hill_1,3) = 0;
A{3}(4,hill_1,4) = 1;
A{3}(5,hill_1,4) = 0;
a{3} = A{3};
% a{2}(:,:,:) = zeros(4,num_states,4);
% a{2}(:,:,1) = a{2}(:,:,1)+1;
% a{2}(:,:,2) = constructLikelihood(a{2}(:,:,2), food_locations, 2);
% a{2}(:,:,2) = constructLikelihood(a{2}(:,:,2), water_locations, 3);
% a{2}(:,:,2) = constructLikelihood(a{2}(:,:,2), sleep_locations, 4);

% a{3}(:,:) = zeros(1,num_states);
% a{3}(hill,1) = 1;
temperature = 0;
%long_term_memory(:,:,:,:,:,:) = ones(35,35,35,5,100);

D{1} = zeros(1,num_states)'; %position in environment
D{2} = [0.25,0.25,0.25,0.25]';

%D{1} = D{1}/numel(D{1});
D{1}(11) = 1;
%D{1} = zeros(25,1);
%D{1}(1,1) = 0.2;
check_hill_visited()
survival(:) = zeros(1,70);
% y{1} = a{1};
% y{2} = softmax(a{2});

 maximum_entropy = maximumEntropy(a{2});
 bucket_size = round(maximum_entropy/14);

D{1} = normalise(D{1});
in_swamp = 0;
% %%% Hyper Params %%%
%  true_food_source_1 = 31
%  true_food_source_2 = 47
%  true_food_source_3 = 24
% true_water_source_1 =10 
% true_water_source_2 = 78
% true_sleep_source_1 =37
% true_sleep_source_2 = 77


food_cutoff = 11;
water_cutoff = 8;
sleep_cutoff = 6;


global num_policies;
num_policies = 2;
num_factors = 1;
T = 14;
num_modalities = 3;
num_iterations = 50;
TimeConst = 4;
num_states = 25;
num_states_low = 25;

short_term_memory(:,:,:) = zeros(100,100);

RL_state_belifs = zeros(T,num_states);
G = zeros(5);
posterior_beta = 1;
gamma(1) = 1/posterior_beta; % expected free energy precision
beta = 1;



E = ones(num_policies,1);


%%% Distributions %%%


for action = 1:5
    B{1}(:,:,action)  =  eye(num_states);
    B{2}(:,:,action)  =  zeros(4);
    B{2}(:,:,action) = [0.8,   0,     0     0.2;
               0.2,   0.8,   0,    0;
               0,     0.2,   0.8,  0;
               0,     0,     0.2   0.8 ];
    
    % Uniform prior over season transitions. This is what the agent must
    % learn
    b{2}(:,:,action) = [  0.25,  0.25,     0.25     0.25;
               0.25,     0.25,     0.25,    0.25;
               0.25,     0.25,     0.25,    0.25;
               0.25,     0.25,       0.25     0.25]; 
end
for i = 1:num_states
    if i ~= [1,6,11,16,21]
        B{1}(:,i,2) = circshift(B{1}(:,i,2),-1); % move left
    end  
end

for i = 1:num_states
    if i ~= [5,10,15,20,25]
        B{1}(:,i,3) = circshift(B{1}(:,i,3),1); % move right
    end  
end

for i = 1:num_states
    if i ~= [21,22,23,24,25]
        B{1}(:,i,4) = circshift(B{1}(:,i,4),5); % move rup
    end  
end

for i = 1:num_states
    if i ~= [1,2,3,4,5]
        B{1}(:,i,5) = circshift(B{1}(:,i,5),-5); % move down
    end  
end


           
% b{2}(:,:) = [ 0.6283,      0.1276,     0.1068,     0.1423;
%                0.1536,     0.2891,     01986,    0.2028;
%                0.1093,     0.3189,     0.4124,    0.2751;
%                0.1087,     0.2643,      0.2822,     0.3799]; 
%b{2}(:,:) = ones(4,4);
%b{2} = b{2}/10;
%b{2} = B{2};
b{1} = B{1};
C{1} = ones(11,9); % preference for positional observation. Uniform.
C_overall{1} = zeros(T,9);

global viable_policies
viable_policies = ones(num_policies,T);

chosen_action = zeros(1,T-1);
preference_values = zeros(4,T);

for factor = 1:num_factors
    NumStates(factor) = size(B{factor},1);   % number of hidden states
    NumControllable_transitions(factor) = size(B{factor},3); % number of hidden controllable hidden states for each factor (number of B matrices)
end


%initially, the prior over states is just a uniform distribution for each
%state factor

for policy = 1:num_policies
    for factor = 1:num_factors
        state_posterior{factor} = ones(numel(D{factor}),T,policy)/numel(D{factor});
    end
end

time_since_food = 0;    
time_since_water = 0;
time_since_sleep = 0;

deterministic_nodes = java.util.Stack();
stochastic_nodes = java.util.Stack();
t = 1;
used_memory_count = 0;
surety = 1;
simulated_time = 0;
memory_count = 0;
for trial = 1:100
while(t<50 && time_since_food < 8)
    unviable_action_count=0;
    remembered_count = 0;
    disp(t)
    prefs = determineObservationPreference(time_since_food);
    bb{2} = normalise_matrix(b{2});
    for factor = 1:2
        if t == 1
            P{t,factor} = D{factor}';
            Q{t,factor} = D{factor}';
            true_states(1, t) = 11;
            true_states(2, t) = find(cumsum(D{2}) >= rand,1);
        else
      %       P{t} = B{factor}(:,higher_level_state, higher_level_action);
            if factor == 1
                %b = B{1}(:,:,chosen_action(t-1));
                Q{t,factor} = (B{1}(:,:,chosen_action(t-1))*Q{t-1,factor}')';
                %Q{t,factor} = Q{t,factor}';
                true_states(factor, t) = find(cumsum(B{1}(:,true_states(factor,t-1),chosen_action(t-1)))>= rand,1);
            else
                %b = B{2}(:,:,:);
                Q{t,factor} = (bb{2}(:,:,chosen_action(t-1))*Q{t-1,factor}')';%(B{2}(:,:)'
                true_states(factor, t) = find(cumsum(B{2}(:,true_states(factor,t-1),1))>= rand,1);   
                 
            end
        end
         if true_states(factor,t) == hill_1 && hill_1_visited==0
                hill_1_visited = 1;
                short_term_memory(:,:) = 0;
                memory_count = 0;
         end
         

        if t > 1
            prev_time_since_food = time_since_food;
%             prev_time_since_water = time_since_water;
%             prev_time_since_sleep = time_since_sleep;
        end
        
        if time_since_food > 99
            time_since_food = 99;
        end
%         if time_since_water > 34
%             time_since_water = 34;
%         end
%         if time_since_sleep > 34
%             time_since_sleep = 34;
%         end
    end
    
    if (true_states(2,t) == 1 && true_states(1,t) == true_food_source_1) || (true_states(2,t) == 2 && true_states(1,t) == true_food_source_2) || (true_states(2,t) == 3 && true_states(1,t) == true_food_source_3) || (true_states(2,t) == 4 && true_states(1,t) == true_food_source_4)
            time_since_food = 0;
%             time_since_water = time_since_water +1;
%             time_since_sleep = time_since_sleep +1;
%                        
%     elseif (true_states(2,t) == 1 && true_states(1,t) == true_water_source_1) || (true_states(2,t) == 2 && true_states(1,t) == true_water_source_2) || (true_states(2,t) == 3 && true_states(1,t) == true_water_source_3) || (true_states(2,t) == 4 && true_states(1,t) == true_water_source_4)
%         time_since_water = 0;
%         time_since_food = time_since_food +1;
%         time_since_sleep = time_since_sleep +1;
% 
%     elseif (true_states(2,t) == 1 && true_states(1,t) == true_sleep_source_1) || (true_states(2,t) == 2 && true_states(1,t) == true_sleep_source_2) || (true_states(2,t) == 3 && true_states(1,t) == true_sleep_source_3) || (true_states(2,t) == 4 && true_states(1,t) == true_sleep_source_4)
%         time_since_sleep = 0;
%         time_since_food = time_since_food +1;
%         time_since_water = time_since_water +1;
      
    else
        if t > 1
            time_since_food = time_since_food +1;
             time_since_water = time_since_water +1;
             time_since_sleep = time_since_sleep +1;
        end

    end
    % sample the next observation. Same technique as sampling states
    
    for modality = 1:num_modalities     
        ob = A{modality}(:,true_states(1,t),true_states(2,t));
        observations(modality,t) = find(cumsum(A{modality}(:,true_states(1,t),true_states(2,t)))>=rand,1);
        %create a temporary vectore of 0s
        vec = zeros(1,size(A{modality},1));
        % set the index of the vector matching the observation index to 1
        vec(1,observations(modality,t)) = 1;
        O{modality,t} = vec;
    end
    true_t = t;
    if t > 1
       
    trajectory_history = [];
    
      start = t - 5;
    if start <= 0
        start = 1;
    end
    qq = P;
    novelty = 0;
    bb{2} = normalise_matrix(b{2});
    
    %Backwards pass to calculate retrospective model (propagated parameter
    %belief search. In this implementation, the agent does it over transition only)
    post = calculate_posterior(Q,A,O,t);
    for timey = start:t
        L = spm_backwards(O,post,A,bb,chosen_action,timey,t);
        LL{timey} = L;
        
%         a_prior  = a{2};
%          for modality = 2:2
%            a_learning = O(modality,timey)';
%            for  factor = 1:num_factors
%                a_learning = spm_cross(a_learning, LL{factor});
%            end
%            a_learning = a_learning.*(a{modality} > 0);
%            a{modality} = a{modality} + 100*a_learning;
%          end
        if timey > start
            b_learning = LL{timey};
            b_learning = spm_cross(b_learning, LL{timey-1});
            b_learning = b_learning.*(b{2} > 0);
            b_prior = b{2};
            b{2} = b{2} + 0.2*b_learning;


            bb{1} = B{1};
            w = kldir(normalise_matrix(b_prior(:,:,1)),normalise_matrix(b{2}(:,:,1)));
            novelty = novelty + w*100;
        end
%         for modality = 2:2
%             a_prior{modality} = a{modality}(:,:,:);
%             a_complexity{modality} = spm_wnorm(a_prior{modality});
%             a_complexity{modality} = a_complexity{modality}.*a_prior{modality};
%         end
%         a1 =a{2};
%         a1 = a1(:);
%          
%         a2 = a_prior;
%         a2 = a2(:);
%          
%          w = kldir(normalise(a2(:)),normalise(a1(:)));
         
       
    end
    end
     bb{2} = normalise_matrix(b{2}(:,:));
     bb{1} = b{1};
      if true_states(2,t) == 1
           food = true_food_source_1;
%           water = true_water_source_1;
%           sleep = true_sleep_source_1;
      elseif true_states(2,t) == 2
          food = true_food_source_2;
%           water = true_water_source_2;
%           sleep = true_sleep_source_2;
      elseif true_states(2,t) == 3
          food = true_food_source_3;
%           water = true_water_source_3;
%           sleep = true_sleep_source_3;
      else
          food = true_food_source_4;
%           water = true_water_source_4;
%           sleep = true_sleep_source_4;
      end
    displayGridWorld(true_states(1,t),food, hill_1, 1)
    g = {};
    % Unused in this iteration, as the agent does not need to learn
    % likelihood
    y{2} = normalise_matrix(a{2});
    y{1} = A{1};
    y{3} = A{3};
    
    prefs = determineObservationPreference(time_since_food);
    horizon = 5;

    long_term_memory =0;
    trajectory = [];
    a_complexity = 0;
 
    current_state = find(cumsum(Q{t,1})>=rand,1)*find(cumsum(Q{t,2})>=rand,1);
    short_term_memory(:,:) = 0;
    
    % Start tree search from current time point
    [G,Q, D, short_term_memory, long_term_memory, traj_count] = tree_search_frwd(long_term_memory, short_term_memory, O, Q ,a, A,A, D, B,b, t, T, t+horizon, time_since_food, resource_locations, current_state, true_t, chosen_action, a_complexity, traj_count, surety, simulated_time, time_since_food, trajectory,0);
    
    u = softmax(G);
    alpha = 1;
    if surety < 0.01
        alpha = surety;
    end

    [maxi, chosen_action(t)] = max(G);
    
   
    t = t+1;
    % end loop over time points

end
survival(trial) = t;
if(numel(true_states) == 18)
    alive_status = 1;
else 
    alive_status = 0;
end

% t_pref_mv_av = movmean(pref_match, 9);
% t_food_mv_av = movmean(t_food_plot,9);
% t_water_mv_av = movmean(t_water_plot,9);
% t_sleep_mv_av = movmean(t_sleep_plot,9);
% fid =fopen('results.txt', 'w' );
% fwrite(fid, 'true_states: ');
% fprintf(fid, '%g,', true_states);
% fprintf(fid, '%g\n','');
% fwrite(fid, 'food_mov_av: ');
% fprintf(fid, '%g,', t_food_mv_av);
% fprintf(fid, '%g\n','');
% fwrite(fid, 'water_mov_av: ');
% fprintf(fid, '%g,', t_water_mv_av);
% fprintf(fid, '%g\n','');
% fwrite(fid, 'sleep_mov_av: ');
% fprintf(fid, '%g,', t_sleep_mv_av);
% fprintf(fid, '%g\n','');
% fwrite(fid, 'overall_mov_av: ');
% fprintf(fid, '%g,', t_pref_mv_av);
% fprintf(fid, '%g\n','');

% fclose(fid);
t = 1;
time_since_food = 0;
end



function [G,P, D, short_term_memory, long_term_memory,traj_count] = tree_search_frwd(long_term_memory, short_term_memory, O, P, a, A,y, D, B,b, t, T, N, t_food, resource_locations, current_state, true_t, chosen_action, novelty, traj_count, surety, simulated_time, true_t_food, trajectory, hill_visited)
    global counti
    global hill_1;
    global remembered_count
    global hill_1_visited;
    global hill_2_visited;
    global hill_3_visited;
    global memory_count;
    
    global unviable_action_count;
    counti = counti+1;

    if hill_1_visited == 1
        local_hill_1_visited = 1;
    else
        local_hill_1_visited = 0;
    end
   
    num_factors = 2;
    G = [0.2,0.2,0.2,0.2,0.2];
    context_prior = P{t,2};
    P = calculate_posterior(P,y,O,t);
    context_posterior = P{t,2};
    total = spm_cross(P{t,:});
    start = t - 5;
    if start <= 0
        start = 1;
    end
    qq = P;
    novelty = 0;
    bb{2} = normalise_matrix(b{2});
    
    for timey = start:t
        L = spm_backwards(O,qq,A,bb,chosen_action,timey,t);
        LL{timey} = L;
        
%         a_prior  = a{2};
%          for modality = 2:2
%            a_learning = O(modality,timey)';
%            for  factor = 1:num_factors
%                a_learning = spm_cross(a_learning, LL{factor});
%            end
%            a_learning = a_learning.*(a{modality} > 0);
%            a{modality} = a{modality} + 100*a_learning;
%          end
        if timey > start
            b_learning = LL{timey-1};
            b_learning = spm_cross(LL{timey}', b_learning');
            b_prior = b{2};
            b{2} = b{2} + b_learning;
            bb{2} = normalise_matrix(b{2});
            bb{1} = B{1};
            w = kldir(normalise_matrix(b_prior(:,:,1)),bb{2}(:,:,1));

    %         for modality = 2:2
    %             a_prior{modality} = a{modality}(:,:,:);
    %             a_complexity{modality} = spm_wnorm(a_prior{modality});
    %             a_complexity{modality} = a_complexity{modality}.*a_prior{modality};
    %         end
    %         a1 =a{2};
    %         a1 = a1(:);
    %          
    %         a2 = a_prior;
    %         a2 = a2(:);
    %          
    %          w = kldir(normalise(a2(:)),normalise(a1(:)));
             novelty = novelty + w;
        end
    end
    if t == true_t
        traj_count(t) = 0;  
    end
    
    if t > true_t
        t_food = t_food*(1-O{2,t}(2));
%         t_water = t_water*(1-O{2,t}(3));
%         t_sleep = t_sleep*(1-O{2,t}(4));
%         t_food = t_food*(1-(P{t}(resource_locations(1)) + P{t}(resource_locations(2)) + P{t}(resource_locations(3))));
        %t_water = t_water*(1-(P{t}(resource_locations(4)) + P{t}(resource_locations(5))));
       % t_sleep = t_sleep*(1-P{t}(resource_locations(6)));
    end
    if t_food>100    
        t_food = 100;
    end


    actions = [1,2,3,4,5];

   
    qs = spm_cross(P{t,:});
    n = novelty(:,:);
    qs = qs(:);
    
    for action = actions
        t_food_approx = round(t_food+1); 
        % Get distribution over next states for both state factors (agent
        % position and season/context)
        Q{1,action} = (B{1}(:,:,action)*P{t,1}')';
        Q{2,action} = (bb{2}(:,:,action)*P{t,2}')';%(B{2}(:,:)*P{t,2}')';
        s = Q(:,action);
        qs = spm_cross(s);
        ambiguity =0;
        next_state(1) = find(cumsum(Q{1,action})>=rand,1);
        next_state(2) = find(cumsum(Q{2,action})>=rand,1);
        % Add epistemic term (see EFE equation)
        epi = epistemic(A,Q(:,action));
        G(action) = G(action) - epi;      
        qs = qs(:);
        % Add novelty to term (see EFE equation)
         G(action) = G(action) + novelty*1000;
        for modality = 2:2
            predictive_observations_posterior = y{2}(:,:)*qs; 
            if modality == 2
                C = determineObservationPreference(t_food_approx);
                %reduce preference precision
                C{modality} = C{modality}/10;
                C = softmax(C{modality});
                ttt=1; 
            end
            if modality == 2
                % add extrinsic term (see EFE equation)
                extrinsic = predictive_observations_posterior'*nat_log(C)';
                G(action) = G(action) + extrinsic;
            end
        end
    end

    if t < N
        action_values = softmax(G(:));
        unviable_actions = action_values <= 1/16;
        G(unviable_actions) = -400;
        actions = [1,2,3,4,5];
        % for each action
        for choices = 1:5
            randomIndex = randi(length(actions), 1);
            action = actions(randomIndex);
            actions = actions(actions~=action); 
            % get distribution over next possible states given that action
            s = Q(:,action);
            qs = spm_cross(s);
            qs = qs(:);
            % only consider relatively likely states
            likely_states = find(qs > 1/16);
            if isempty(likely_states)
                threshold = 1/numel(qs)*1/numel(qs);
                likely_states = find(qs > (1/numel(qs)-threshold));
            end
            % for each of those likely states
            for state = likely_states(:)'
                % check to see if we have already calculated a value for
                % this state
                if short_term_memory(t_food_approx, state) ~= 0 
                    sh = short_term_memory(t_food_approx,state);
                    S =  sh;  
                else
                    % get distribution over possible observations given
                    % state
                    for modal = 1:numel(A) 
                      O{modal,t+1} = normalise(y{modal}(:,state)');
                    end   
                    % prior over next states given transition function
                    % (calculated earlier)
                    P{t+1,1} = Q{1, action};
                    P{t+1,2} = Q{2, action};
                    traj_count(true_t) = traj_count(true_t)+1;   
                    traj = trajectory;
                    chosen_action(t) = action;
                    % recursively move to the next node (likely state) of
                    % the tree
                    [expected_free_energy, d, D, short_term_memory, long_term_memory, traj_count] = tree_search_frwd(long_term_memory, short_term_memory, O, P, a, A,y, D, B,b, t+1, T, N, t_food_approx, resource_locations, state, true_t, chosen_action, novelty, traj_count, surety, 0, true_t_food, traj, hill_visited);
                    S = max(expected_free_energy);
                    short_term_memory(t_food_approx,state) = S;
               end
                K(state) = S;
            end
            action_fe = K(likely_states)*qs(likely_states);
            G(action) = G(action) + 0.7*action_fe;
        end
    else
    end
       
            
          
    
     
end
function P = calculate_posterior(P,A,O,t)
for fact = 2:2
        L     = 1;
        num = numel(A);
        for modal = 2:num
                obs = find(cumsum(O{modal,t})>= rand,1);
                temp = A{modal}(obs,:,:);
                temp = permute(temp,[3,2,1]);
                L = L.*temp;
        end
        %L = permute(L,[3,2,1]);
        for f = 1:2
            if f ~= fact
                if f == 2
                    LL = P{t,f}*L;
                else
                    LL = L*P{t,f}';
                end
            end
        end
        y = LL.*P{t,fact}';
        P{t,fact}  = normalise(y)';       
end
end


function action = selectAIAction(actions, temp)
action_choices = [1,2,3,4,5];
[M,I] = max(actions); 
max_actions = find(actions == M);
    max_action = max_actions(randsample(numel(max_actions), 1));
temp = temp*1000;
epsilon = randsample(1000,1);

if epsilon <= temp
    random_actions = action_choices(find(action_choices~=I));
    action = random_actions(randsample(numel(random_actions),1));
else
    action = max_action;
end
end


function updateQValues(observation_prev, context, observation, reward, action)
global Q_table;
current_Q_value = Q_table(action, context,observation_prev);
next_Q_value = max(Q_table(:,context,observation));
Q_table(action,context, observation_prev) = Q_table(action,context, observation_prev)  + 0.3*(reward + 0.5*next_Q_value - current_Q_value);
end



function y = nat_log(x)
y = log(x+exp(-500));
end 


function horizon = determineLookAhead(t_food, t_water, t_sleep)
horizon = min([t_food, t_water, t_sleep]);
end



function C = determineObservationPreference(t_food)

% if t_food > 13
%     t_water = -50;
%     t_sleep = -50;
%     empty = -50;
% end
% % if t_water >13  
% %     t_food = t_food-50;
% %     t_sleep = t_sleep-50;
% %     empty = -50;
% %     
% % end
% % if t_sleep >13
% %     t_food =t_food -50;
% %     t_water =t_water -50;
% %     empty = -50;
% % end
% mini = min([t_food, t_water, t_sleep]);
% maxi = max([t_food, t_water, t_sleep]);
% empty_preference = -exp(mini)/4;
% food_preference = exp(t_food)-exp(maxi)/4;
% water_preference = exp(t_water)-exp(maxi)/4;
% sleep_preference = exp(t_sleep)-exp(maxi)/4;
% % C{2} = preferences for [empty, food, water, sleep]
% C{2} = [empty_preference ,food_preference, water_preference, sleep_preference];
empty = -1 ;
% it really wants to avoid going 11 timesteps without food, similarly for
% sleep and water. If it does, all other types of squares become hugely
% negative in reward
%  if t_food == 0
%     empty = 1;
% else 
%     empty = 1;
% end
if t_food > 7
     
    empty = -500;

end
%true_average = (t_food + t_water + t_sleep)/3;
%maxi = max([t_food, t_water, t_sleep]);
average = t_food/2 ;
%C{2} = [empty, t_food-average, t_water-average, t_sleep-average];
% else
%     C{2} = [1,1,1,1];
% end


C{2} = [empty, t_food];

end

function x = normalise(array)
x = array/(sum(array));
if isnan(x)
    x = ones(numel(x),1);
    x(:) = 1/numel(x);
end
end

function m = normalise_matrix(m)
  for i = 1:length(m(1,:))
      m(:,i) = m(:,i)/sum(m(:,i));
  end
end

function b = B_norm(B)
bb = B; 
z = sum(bb,1); %create normalizing constant from sum of columns
bb = bb./z; % divide columns by constant
bb(isnan(bb)) = 0; %replace NaN with zero
b = bb; 
% insert zero value condition
end 

function S = softmax(v)
%S = exp(v)/sum(exp(v));
 S = bsxfun(@rdivide,exp(v),sum(exp(v)));
end

% epistemic value term (Bayesian surprise) in expected free energy 
function epi = epistemic(A,s)
    epic = 0;
    qs = spm_cross(s); 
    qs = qs(:);
    epi = 0;
    for modal = 1:numel(A)
        gg = A{modal}(:,:);
        states = find(qs > exp(-16));
        for ss = 1:numel(states)
            state = states(ss);
            for obs = 1:numel(gg(:,1))
                qo = gg(obs,:)*qs;
                aa = nat_log(gg(obs,state));
                aaa = nat_log(qo);
                epi = epi + qs(state)*gg(obs,state)*(aaa-aa);
            end
        end
        t = 1;        
    end
           
end



function [Y] = spm_cross(X,x,varargin)
% Multidimensional outer product
% FORMAT [Y] = spm_cross(X,x)
% FORMAT [Y] = spm_cross(X)
%
% X  - numeric array
% x  - numeric array
%
% Y  - outer product
%
% See also: spm_dot
% Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_cross.m 7527 2019-02-06 19:12:56Z karl $

% handle single inputs
if nargin < 2
    if isnumeric(X)
        Y = X;
    else
        Y = spm_cross(X{:});
    end
    return
end

% handle cell arrays

if iscell(X), X = spm_cross(X{:}); end
if iscell(x), x = spm_cross(x{:}); end

% outer product of first pair of arguments (using bsxfun)
A = reshape(full(X),[size(X) ones(1,ndims(x))]);
B = reshape(full(x),[ones(1,ndims(X)) size(x)]);
Y = squeeze(bsxfun(@times,A,B));

% and handle remaining arguments
for i = 1:numel(varargin)
    Y = spm_cross(Y,varargin{i});
end
end

function amb = ambiguity(likelihood, state)
amb = 0;

for obs = 1:4       
    amb = amb + likelihood(obs,state)*nat_log(likelihood(obs,state));
end        
   
end

function a = displayGridWorld(agent_position, food_position_1,hill_1_pos,alive_status)
if alive_status == 1
    agent_text = 'A';
else 
    agent_text = 'Dead';
end

agent_dim1 = 0;
if agent_position <= 5
    agent_dim2 = 1;
    agent_dim1 = agent_position;
elseif agent_position < 11
    agent_dim2 = 2;
    agent_dim1 = agent_position - 5;
elseif agent_position < 16
    agent_dim2 = 3;
    agent_dim1 = agent_position - 10;
elseif agent_position < 21
    agent_dim2 = 4;
    agent_dim1 = agent_position - 15;
elseif agent_position < 51
    agent_dim2 = 5;
    agent_dim1 = agent_position - 20;
elseif agent_position < 61
    agent_dim2 = 6;
    agent_dim1 = agent_position - 50;
elseif agent_position < 71
    agent_dim2 = 7;
    agent_dim1 = agent_position - 60;
elseif agent_position < 81
    agent_dim2 = 8;
    agent_dim1 = agent_position - 70;
elseif agent_position < 91
    agent_dim2 = 9;
    agent_dim1 = agent_position - 80;
else
    agent_dim2 = 5;
    agent_dim1 = agent_position - 20;
end

locations_1 = [];
hill_1_dim2 = idivide(int16(hill_1_pos),5,'floor')+1;
hill_1_dim1 = rem(hill_1_pos,5);
if hill_1_dim1 == 0
    if hill_1_dim2 ~= 1
        hill_1_dim2 = hill_1_dim2-1;
    end
    hill_1_dim1 = 5;
end

food_1_dim2 = idivide(int16(food_position_1),5,'floor')+1;
food_1_dim1 = rem(food_position_1,5);
if food_1_dim1 == 0
    if food_1_dim2 ~= 1
        food_1_dim2 = food_1_dim2-1;
    end
    food_1_dim1 = 5;
end



h1=figure(1);
set(h1,'name','gridworld');
h1.Position = [400 200 800 700];
[X,Y]=meshgrid(1:6,1:6);
plot(Y,X,'k'); hold on; axis off
plot(X,Y,'k');hold off; axis off
hold off;
I=(1);
surface(I);
h=linspace(0.5,1,64);
%h=[h',h',h'];
%set(gcf,'Colormap',h);
q=1;
x=linspace(1.5,5.5,5);
y=linspace(1.5,5.5,5);
%empty_pref =sprintf('%.3f',preference_values(1));
%food_pref =sprintf('%.3f',preference_values(2));
%water_pref =sprintf('%.3f',preference_values(3));
%sleep_pref =sprintf('%.3f',preference_values(4));
for n=1:5
    for p=1:5
        if n == agent_dim1 & p == agent_dim2
            text(y(n)-.2,x(p),agent_text,'FontSize',16);
            q=q+1;
        
        end
        
        if (n == food_1_dim1 & p == food_1_dim2) 
            text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
            %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
            q=q+1;
        end
        
%         if (n == water_1_dim1 & p == water_1_dim2) 
%             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
%             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
%             q=q+1;
%         end
        
        
        if (n == hill_1_dim1 & p == hill_1_dim2)
            text(y(n)-.2,x(p)+.3,'Hill','FontSize',16, 'FontWeight','bold');
            %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
            q=q+1;
        end
       

        
    end
end


end


%--------------------------------------------------------------------------
function A  = spm_wnorm(A)
% This uses the bsxfun function to subtract the inverse of each column
% entry from the inverse of the sum of the columns and then divide by 2.
% 
A   = A + exp(-16);

A   = bsxfun(@minus,1./A, 1./sum(A,1))/2;
%sum_a = sum(A,1);
%a_sums = 1./sum(A,1);
%w = (1./A-a_sums)/2;
%disp(w)
end 


function preferencePlot(preference_plot)
table = array2table(preference_plot,'VariableNames',{'Emtpy','Food','Water','Sleep'},'RowNames',{'Preference','Observation'});
uitable('Data',table{:,:},'ColumnName',table.Properties.VariableNames,...
    'RowName',table.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
end


function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);
end

function novelty = calculate_novelty(W, A, S)
    novelty = 0;
    %for g = 1:numel(A)
        complexity = W*S;
        n=A*S;
        novelty = novelty + n'*complexity;
    %end
end


function horizon = calculateLookAhead(ent, bucket_size, surety)
    if surety < 0.5  
        horizon = 6 + round(14/(ent/bucket_size));
        if horizon > 14
            horizon = 14;
        end
    else
        horizon = 6;
    end
end

function entropy = maximumEntropy(A)
entropy = 0;    
states = numel(A(1,:,1));
L = A(:,:);
for state = 1:states
    state_dist = L(:,state);
    information = log2(state_dist)+exp(-16);
    entropy = entropy + -state_dist'*log2(state_dist)+exp(-16);
end   
end
function relative_entropy = calc_relative_entropy(A,state)
    uniform(:) = [0.25,0.25,0.25,0.25];
    likelihood = A(:,state)+exp(-16);
    information = -log2(likelihood)+exp(-16);
    entropy = likelihood'*information;
    maximum_entropy_information = -log2(uniform);
    maximum_entropy = maximum_entropy_information*uniform';
    relative_entropy = entropy/maximum_entropy;  
    dummy=1;
end

function relative_entropy = calc_entropy(A)
    uniform(:) = [0.25,0.25,0.25,0.25];
    information = -log2(A+exp(-16));
    entropy = A'*information;
    maximum_entropy_information = -log2(uniform);
    maximum_entropy = maximum_entropy_information*uniform';
    relative_entropy = entropy/maximum_entropy;  
    dummy=1;
end
function [pos, previous_positions] = randPos(previous_positions)
pos = randsample(100,1);
while ismember(pos, previous_positions)
    pos = randsample(100,1);
end
previous_positions(end+1) = pos;
end

function a = constructLikelihood(a, resource_positions, type)
for location = 1:length(resource_positions)
    a(type, resource_positions(location)) = 1;
    a(1, resource_positions(location)) = 0;
end      
end

function a = constructLocalHills(a, resource_positions, type, hillpos)
for hill_pos = 1:length(hillpos)
    for i = hillpos(hill_pos)-3:hillpos(hill_pos)+3
        j = i;
        while(true)
            if j-10 > 0 && j-10>i-30
                j = j-10;
            else
                break
            end
        end
            
        while(j<i+30 && j < 100)
            occupied = [];
             for k = 2:4
                if a(k,j) == 1
                    occupied(end+1) = k;
                end
             end
            if ismember(j, resource_positions)
                
                a(type, j) = 1;
                
                for k = 1:4
                    if k ~= type && ~ismember(k, occupied)
                        a(k, j) = 0;
                    end
                end
                
               
            else
                if (j >0 && j<100) && (i > 0 && i < 100) && isempty(occupied)
                    
                   a(1,j) = 1;
                end
            end
            
            j = j+10;
        end
    end
end
end
function a = constructUnknownAreas(a)
for i = 1:length(a(1,:))
    temp = a(:,i);
    if sum(temp) == 0
       a(:,i) = 1;
    end
end
end

function check_hill_visited()
global hill_visited
disp(hill_visited)
end

function n = abs_normalise(array)
abs_array = abs(array);
n = array./abs_array;
end

function d = add_three_matrix_elements(a,b,c)
      d(:,:) = zeros(4,100);
    for i = 1:length(a(:,1))
        for j = 1:length(a(1,:))
            d(i,j) = a(i,j)+ b(i,j) + c(i,j);
            
            if d(i,j) >1
                d(i,j) = 1;
   
            end
        end
    end
end

function d = add_two_matrix_elements(a,b)
    d(:,:) = zeros(4,100);
    for i = 1:length(a(:,1))
        for j = 1:length(a(1,:))
            d(i,j) = a(i,j)+ b(i,j);
            if d(i,j) > 1
                d(i,j) = 1;
            
            end
        end
    end
end

function [L] = spm_backwards(O,Q,A,B,u,t,T)
% Backwards smoothing to evaluate posterior over initial states
%--------------------------------------------------------------------------
L     = Q{t,2};
p     = 1; 
for timestep = (t + 1):T
    
    % belief propagation over hidden states
    %------------------------------------------------------------------
    

    p    = B{2}(:,:,1)*p;
    
    for state = 1:numel(L)
        % and accumulate likelihood
        %------------------------------------------------------------------
        for g = 3:3
           % possible_states = O{g,timestep}*A{g}(:,:);
           obs = find(cumsum(O{g,timestep})>= rand,1);
           temp = A{g}(obs,:,:);
           temp = permute(temp,[3,2,1]);
           temp = temp*Q{timestep,1}';
           aaa = temp'*p(:,state);
           L(state) = L(state).*aaa;
          
        end
    end
end

% marginal distribution over states
%--------------------------------------------------------------------------
L     = spm_norm(L(:));
end


function kl = kldir(a,b)
kl = 0;
for j = 1:numel(a(1,:)) % for each column
    for i = 1:numel(a(:,j)) % for each row
        loga = log(a(i,j));
        logb = log(b(i,j));
       kl =  kl + a(i,j) * (loga - logb);
    end
end
end










        
        
   



   
   