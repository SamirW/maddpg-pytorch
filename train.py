import torch
import os
import numpy as np
from utils.buffer import ReplayBuffer
from torch.autograd import Variable
from gym.spaces import Box


def train(maddpg, env, replay_buffer, config, log, logger, run_dir, log_dir):
    t = 0
    flip = False

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        ep_rew = 0.

        # Flip episodes          
        if ep_i == config.flip_ep:
            print("[ INFO ] Flipping environment")
            replay_buffer = ReplayBuffer(
                config.buffer_length, 
                maddpg.nagents,
                [obsp.shape[0] for obsp in env.observation_space],
                [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space])
            flip = True

        obs = env.reset(flip=flip)
        maddpg.prep_rollouts(device='cpu')

        # Reset noise
        if ep_i >= config.flip_ep:
            explr_pct_remaining = max(0, 2 * (config.n_exploration_eps - (ep_i - config.flip_ep))) / config.n_exploration_eps
        else:
            explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps

        maddpg.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            torch_obs = [
                Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                for i in range(maddpg.nagents)]

            # get actions
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            # Take step
            next_obs, rewards, dones, infos = env.step(actions)
            # if et_i == config.episode_length - 1:
            #     dones = dones + 1

            # Push to replay buffer
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            # For next step
            obs = next_obs
            t += config.n_rollout_threads
            ep_rew += rewards[0][0]

            # Get MADDPG critic values for debugging
            critics = maddpg.get_critic_vals(torch_obs, torch_agent_actions)
            logger.add_scalar('debug/critic0', critics[0][0].detach().numpy()[0], ep_i)
            logger.add_scalar('debug/critic1', critics[1][0].detach().numpy()[0], ep_i)

            # MADDPG update
            if ep_i < config.eval_ep:
                if len(replay_buffer) >= config.batch_size:
                    if t % config.steps_per_update == 0:
                        maddpg.prep_training(device='cpu')

                        for u_i in range(config.n_rollout_threads):
                            for a_i in range(maddpg.nagents):
                                sample = replay_buffer.sample(config.batch_size, to_gpu=False)
                                if ((ep_i > config.flip_ep) and (ep_i < (config.flip_ep + config.skip_actor_length))):
                                    maddpg.update(sample, a_i, logger=logger, skip_actor=True)
                                else:
                                    maddpg.update(sample, a_i, logger=logger)
                            maddpg.update_all_targets()
                        maddpg.prep_rollouts(device='cpu')

        logger.add_scalar('joint/mean_episode_rewards', ep_rew, ep_i)
        log[config.log_name].info("Train episode reward {:0.5f} at episode {}".format(ep_rew, ep_i))

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            maddpg.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            maddpg.save(str(run_dir / 'model.pt'))

        if (ep_i + 1) == config.hard_distill_ep:
            print("************Distilling***********")
            os.makedirs(str(run_dir / 'models'), exist_ok=True)
            maddpg.save(str(run_dir / 'models/before_distillation.pt'.format(ep_i)))

            maddpg.prep_rollouts(device='cpu')
            maddpg.distill(256, 1024, replay_buffer, hard=True)

            maddpg.save(str(run_dir / 'models/after_distillation.pt'.format(ep_i)))

        if (ep_i) % config.model_save_freq == 0:
            print("************Saving model***********")
            os.makedirs(str(run_dir / 'models'), exist_ok=True)
            maddpg.save(str(run_dir / 'models/model{}.pt'.format(ep_i)))
    import sys
    sys.exit()

    # print("***********Resettting************")
    # maddpg.agents[0].reset()

    # Save experience replay buffer
    if config.save_buffer:
        print("*******Saving Replay Buffer******")
        import pickle 
        with open(str(run_dir / 'replay_buffer.pkl'), 'wb') as output:
            pickle.dump(replay_buffer, output, -1)

    print("********Saving and Closing*******")
    maddpg.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
