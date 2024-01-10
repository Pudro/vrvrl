import argparse
import tensorflow as tf
import numpy as np
import sys
import collections
import datetime
import pickle

def train(config: argparse.Namespace):
# ===== Training History ======
    history_distances = []
    history_steps = []
    history_loss = []
    history_solutions = []

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        policy_estimator = PolicyEstimator()
        initialize_uninitialized(sess)
        print(sess.run(tf.report_uninitialized_variables()))
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable={}, Shape={}".format(k, v.shape))
        sys.stdout.flush()
        saver = tf.train.Saver()
        if config.model_to_restore is not None:
            saver.restore(sess, config.model_to_restore)

        distances = []
        steps = []
        consolidated_distances, consolidated_steps = [], []
        timers = []
        num_checkpoint = int(config.max_rollout_steps/config.step_interval)
        step_record = np.zeros((2, num_checkpoint))
        distance_record = np.zeros((2, num_checkpoint))
        start = datetime.datetime.now()
        seed = config.problem_seed
        tf.set_random_seed(seed)

        Transition = collections.namedtuple("Transition", ["state", "trip", "next_distance", "action", "reward", "next_state", "done"])
        for index_sample in range(config.num_episode):
            states = []
            trips = []
            actions = []
            advantages = []
            action_labels = []
            if index_sample > 0 and index_sample % config.debug_steps == 0:
                if not config.use_random_rollout:
                    formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                    count_timers = formatted_timers[4:][::2]
                    time_timers = formatted_timers[4:][1::2]
                    print('time ={}'.format('\t\t'.join([str(x) for x in time_timers])))
                    print('count={}'.format('\t\t'.join([str(x) for x in count_timers])))
                    start_active = ((len(distances) // config.num_active_learning_iterations) * config.num_active_learning_iterations)
                    if start_active == len(distances):
                        start_active -= config.num_active_learning_iterations
                    tail_distances = distances[start_active:]
                    tail_steps = steps[start_active:]
                    min_index = np.argmin(tail_distances)
                    if config.num_active_learning_iterations == 1 or len(distances) % config.num_active_learning_iterations == 1:
                        consolidated_distances.append(tail_distances[min_index])
                        consolidated_steps.append(tail_steps[min_index] + min_index * config.max_rollout_steps)
                    else:
                        consolidated_distances[-1] = tail_distances[min_index]
                        consolidated_steps[-1] = tail_steps[min_index] + min_index * config.max_rollout_steps
                    print('index_sample={}, mean_distance={}, mean_step={}, tail_distance={}, last_distance={}, last_step={}, timers={}'.format(
                        index_sample,
                        format_print(np.mean(consolidated_distances)), format_print(np.mean(consolidated_steps)),
                        format_print(np.mean(consolidated_distances[max(0, len(consolidated_distances) - 1000):])),
                        format_print(consolidated_distances[-1]), consolidated_steps[-1],
                        formatted_timers[:4]
                    ))
                    sys.stdout.flush()
                else:
                    formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                    for index in range(num_checkpoint):
                        print('rollout_num={}, index_sample={}, mean_distance={}, mean_step={}, last_distance={}, last_step={}, timers={}'.format(
                            (index + 1) * config.step_interval, index_sample, ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample,
                            ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample, distance_record[1, index],
                            step_record[1, index], formatted_timers[:4]
                        ))
                        step_record[0, index] = ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample
                        distance_record[0, index] = ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample
                    sys.stdout.flush()

            problem = generate_problem()
            # print(problem.distance_matrix)
            # print(problem.locations)
            # print(problem.capacities)
            solution = construct_solution(problem)
            best_solution = copy.deepcopy(solution)

            if config.use_attention_embedding:
                embedded_trip = embed_solution_with_attention(problem, solution)
            else:
                embedded_trip = [0]
            min_distance = calculate_solution_distance(problem, solution)
            min_step = 0
            distance = min_distance

            state = env_generate_state()
            env_start_time = datetime.datetime.now()
            episode = []
            current_best_distances = []
            start_distance = distance
            current_distances = []
            start_distances = []

            inference_time = 0
            gpu_inference_time = 0
            env_act_time = 0
            no_improvement = 0
            loop_action = 0
            num_random_actions = 0
            for action_index in range(len(action_timers)):
                action_timers[action_index] = 0.0
            for step in range(config.max_rollout_steps):
                start_timer = datetime.datetime.now()
                if config.use_cyclic_rollout:
                    choices = [1, 3, 4, 5, 8]
                    if no_improvement == len(choices) + 1:
                        action = 0
                        no_improvement = 0
                    else:
                        action = choices[loop_action]
                        loop_action += 1
                        if loop_action == len(choices):
                            loop_action = 0
                elif config.use_random_rollout:
                    action = random.randint(0, config.num_actions - 1)
                else:
                    gpu_start_time = datetime.datetime.now()
                    action_probs = policy_estimator.predict([state], [embedded_trip], sess)
                    gpu_inference_time += (datetime.datetime.now() - gpu_start_time).total_seconds()
                    action_probs = action_probs[0]
                    history_action_probs = np.zeros(len(action_probs))
                    action_prob_sum = 0.0
                    for action_prob_index in range(len(action_probs)):
                        action_prob_sum += action_probs[action_prob_index]
                    for action_prob_index in range(len(action_probs)):
                        action_probs[action_prob_index] /= action_prob_sum
                    if config.use_history_action_distribution and (index_sample > 0):
                        history_action_count_sum = 0
                        for action_count_index in range(len(action_probs)):
                            history_action_count_sum += count_timers[action_count_index + 1]
                        for action_count_index in range(len(action_probs)):
                            history_action_probs[action_count_index] = count_timers[action_count_index + 1]/history_action_count_sum
                            action_probs[action_count_index] = action_probs[action_count_index]/2 + history_action_probs[action_count_index]/2


                    if config.use_rl_loss:
                        states.append(state)
                        trips.append(embedded_trip)
                    elif random.uniform(0, 1) < 0.05:
                        action_label = [0] * config.num_actions
                        action_index = 0
                        min_action_time = sys.maxint
                        rewards = []
                        action_times = []
                        for action_to_label in range(1, config.num_actions):
                            action_start_time = datetime.datetime.now()
                            _, reward, _, _, _ = env_step(step, state, problem, min_distance, solution, distance, action_to_label, False)
                            action_time = (datetime.datetime.now() - action_start_time).total_seconds()
                            rewards.append(reward)
                            action_times.append(action_time)
                            if reward > EPSILON and action_time < min_action_time:
                                action_index = action_to_label
                                min_action_time = action_time
                                break
                        action_label[action_index] = 1
                        states.append(state)
                        trips.append(embedded_trip)
                        action_labels.append(action_label)

                    if (config.model_to_restore is not None and should_restart(min_distance, distance, no_improvement)) or no_improvement >= config.max_no_improvement:
                        action = 0
                        no_improvement = 0
                    else:
                        if np.random.uniform() < config.epsilon_greedy:
                            action = np.random.randint(config.num_actions - 1) + 1
                            num_random_actions += 1
                        else:
                            if config.sample_actions_in_rollout:
                                action = np.random.choice(np.arange(len(action_probs)), p=action_probs) + 1
                            else:
                                action = np.argmax(action_probs) + 1
                end_timer = datetime.datetime.now()
                inference_time += (end_timer - start_timer).total_seconds()
                start_timer = end_timer

                next_state, reward, done, next_solution, next_distance = env_step(step, state, problem, min_distance, solution, distance, action)
                if next_distance >= distance - EPSILON:
                    no_improvement += 1
                else:
                    #TODO
                    no_improvement = 0

                current_distances.append(distance)
                start_distances.append(start_distance)
                if action == 0:
                    start_distance = next_distance
                current_best_distances.append(min_distance)
                if next_distance < min_distance - EPSILON:
                    min_distance = next_distance
                    min_step = step
                    best_solution = copy.deepcopy(next_solution)
                if (step + 1) % config.step_interval == 0:
                    print('rollout_num={}, index_sample={}, min_distance={}, min_step={}'.format(
                        step + 1, index_sample, min_distance, min_step
                    ))
                    temp_timers = np.asarray(action_timers)
                    temp_count_timers = temp_timers[::2]
                    temp_time_timers = temp_timers[1::2]
                    print('time ={}'.format('\t\t'.join([str(x) for x in temp_time_timers])))
                    print('count={}'.format('\t\t'.join([str(x) for x in temp_count_timers])))
                if done:
                    break

                episode.append(Transition(
                    state=state, trip=copy.deepcopy(embedded_trip), next_distance=next_distance,
                    action=action, reward=reward, next_state=next_state, done=done))
                state = next_state
                solution = next_solution
                if config.use_attention_embedding:
                    embedded_trip = embed_solution_with_attention(problem, solution)
                else:
                    embedded_trip = [0]
                distance = next_distance
                end_timer = datetime.datetime.now()
                env_act_time += (end_timer - start_timer).total_seconds()
                start_timer = end_timer

            if config.use_random_rollout:
                temp = np.inf
                for rollout_step in range(num_checkpoint):
                    current_region_min_step = np.argmin(current_distances[(rollout_step * config.step_interval):((rollout_step + 1) * config.step_interval)]) + rollout_step * config.step_interval
                    current_region_min_distance = min(current_distances[(rollout_step * config.step_interval):((rollout_step + 1) * config.step_interval)])
                    if temp > current_region_min_distance:
                        distance_record[1, rollout_step] = current_region_min_distance
                        step_record[1, rollout_step] = current_region_min_step
                        temp = current_region_min_distance
                    else:
                        distance_record[1, rollout_step] = distance_record[1, rollout_step - 1]
                        step_record[1, rollout_step] = step_record[1, rollout_step - 1]

            start_timer = datetime.datetime.now()
            distances.append(min_distance)
            steps.append(min_step)
            if validate_solution(problem, best_solution, min_distance):
                print('solution={}'.format(best_solution))
            else:
                print('invalid solution')
            if not (config.use_cyclic_rollout or config.use_random_rollout):
                future_best_distances = [0.0] * len(episode)
                future_best_distances[-1] = episode[len(episode) - 1].next_distance
                step = len(episode) - 2
                while step >= 0:
                    if episode[step].action != 0:
                        future_best_distances[step] = future_best_distances[step + 1] * config.discount_factor
                    else:
                        future_best_distances[step] = current_distances[step]
                    step = step - 1

                historical_baseline = None
                for t, transition in enumerate(episode):
                    # total_return = sum(config.discount_factor**i * future_transition.reward for i, future_transition in enumerate(episode[t:]))
                    if historical_baseline is None:
                        if transition.action == 0:
                            #TODO: dynamic updating of historical baseline, and state definition
                            historical_baseline = -current_best_distances[t]
                            # historical_baseline = 1/(current_best_distances[t] - 10)
                        actions.append(0)
                        advantages.append(0)
                        continue
                    # if transition.action == 0:
                    #     historical_baseline = -current_distances[t]
                    if transition.action > 0:
                        # total_return = transition.reward
                        if transition.reward < EPSILON:
                            total_return = -1.0
                        else:
                            total_return = 1.0
                        #     total_return = min(transition.reward, 2.0)
                        # total_return = start_distances[t] - future_best_distances[t]
                        # total_return = min(total_return, 1.0)
                        # total_return = max(total_return, -1.0)
                        total_return = -future_best_distances[t]
                        # total_return = 1/(future_best_distances[t] - 10)
                    else:
                        if transition.state[-1] != 0 and transition.state[-2] < 1e-6:
                            # if future_best_distances[t] < current_best_distances[t] - 1e-6:
                            total_return = 1.0
                        else:
                            total_return = -1.0
                        total_return = 0
                        actions.append(0)
                        advantages.append(0)
                        continue
                    # baseline_value = value_estimator.predict(states)
                    # baseline_value = 0.0
                    baseline_value = historical_baseline
                    advantage = total_return - baseline_value
                    actions.append(transition.action)
                    advantages.append(advantage)
                    # value_estimator.update(states, [[total_return]], sess)

                states = np.reshape(np.asarray(states), (-1, env_observation_space_n)).astype("float32")
                if config.use_attention_embedding:
                    trips = np.reshape(np.asarray(trips), (-1, config.num_training_points, config.input_embedded_trip_dim_2)).astype("float32")
                actions = np.reshape(np.asarray(actions), (-1))
                advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")
                if config.use_rl_loss:
                    print('num_random_actions={}'.format(num_random_actions))
                    print('actions={}'.format(actions[:100]).replace('\n', ''))
                    print('advantages={}'.format(advantages[:100]).replace('\n', ''))
                    if config.model_to_restore is None and index_sample <= config.max_num_training_epsisodes:
                        filtered_states = []
                        filtered_trips = []
                        filtered_advantages = []
                        filtered_actions = []
                        end = 0
                        for action_index in range(len(actions)):
                            if actions[action_index] > 0:
                                filtered_states.append(states[action_index])
                                filtered_trips.append(trips[action_index])
                                filtered_advantages.append(advantages[action_index])
                                filtered_actions.append(actions[action_index] - 1)
                            else:
                                num_bad_steps = config.max_no_improvement
                                end = max(end, len(filtered_states) - num_bad_steps)
                                filtered_states = filtered_states[:end]
                                filtered_trips = filtered_trips[:end]
                                filtered_advantages = filtered_advantages[:end]
                                filtered_actions = filtered_actions[:end]
                        filtered_states = filtered_states[:end]
                        filtered_trips = filtered_trips[:end]
                        filtered_advantages = filtered_advantages[:end]
                        filtered_actions = filtered_actions[:end]
                        num_states = len(filtered_states)
                        if config.use_attention_embedding and num_states > config.batch_size:
                            downsampled_indices = np.random.choice(range(num_states), config.batch_size, replace=False)
                            filtered_states = np.asarray(filtered_states)[downsampled_indices]
                            filtered_trips = np.asarray(filtered_trips)[downsampled_indices]
                            filtered_advantages = np.asarray(filtered_advantages)[downsampled_indices]
                            filtered_actions = np.asarray(filtered_actions)[downsampled_indices]
                        loss = policy_estimator.update(filtered_states, filtered_trips, filtered_advantages, filtered_actions, sess)
                        print('loss={}'.format(loss))
                else:
                    #TODO: filter and reshape
                    action_labels = np.reshape(np.asarray(action_labels), (-1, config.num_actions))
                    loss, accuracy = policy_estimator.train(states, trips, action_labels, sess)
                    print('loss={}, accuracy={}'.format(loss, accuracy))
            timers_epoch = [inference_time, gpu_inference_time, env_act_time, (datetime.datetime.now() - start_timer).total_seconds()]
            timers_epoch.extend(action_timers)
            timers.append(timers_epoch)
            if config.model_to_restore is None and index_sample > 0 and index_sample % 500 == 0:
                save_path = saver.save(sess, "./rollout_model_{}_{}_{}.ckpt".format(index_sample, config.num_history_action_use, config.max_rollout_steps))
                print("Model saved in path: %s" % save_path)

            #history_distances.append(min_distance)
            history_distances = distances
            history_steps.append(min_step)
            history_loss.append(loss)
            history_solutions.append(solution)

        # save_path = saver.save(sess, "./rollout_model.ckpt")
        # print("Model saved in path: %s" % save_path)
        print('solving time = {}'.format(datetime.datetime.now() - start))

        history_data = {
            'distances': history_distances,
            'min_distance': min_distance,
            'steps': history_steps,
            'loss': history_loss,
            'solving_time': datetime.datetime.now() - start,
            'num_points': config.num_training_points,
            'num_episodes': config.num_episode,
            'solutions': history_solutions,
            'best_solution': best_solution,
            'problem': problem
        }

        with open(f'./results/vrp_{config.num_training_points}_e_{config.num_episode}.pkl', 'wb') as f:
            pickle.dump(history_data, f)

    raise NotImplementedError
