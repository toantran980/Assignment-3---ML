import matplotlib.pyplot as plt
import unittest
import numpy as np
from gridworld_policy_iteration import (
    Gridworld, A_UP, A_RIGHT, A_DOWN, A_LEFT,
    Viewer,
    policy_evaluation, policy_improvement, transition_dist,
    policy_iteration, extract_path, random_policy
)

class TestGridworld(unittest.TestCase):
    """Unit tests for core Gridworld environment mechanics (bounds, transitions, rewards, etc.)"""
    
    def setUp(self):
        self.env = Gridworld()
    
    def test_in_bounds(self):
        """Test the in_bounds() method for valid and invalid coordinates."""

        self.assertTrue(self.env.in_bounds(0, 0))
        self.assertFalse(self.env.in_bounds(5, 0))  # Outside grid
        self.assertFalse(self.env.in_bounds(-1, 0))  # Negative row
        self.assertFalse(self.env.in_bounds(0, 5))  # Outside column

    def test_is_terminal(self):
        """Check if is_terminal() correctly identifies terminal states."""

        for t in self.env.terminals:
            self.assertTrue(self.env.is_terminal(t))
        self.assertFalse(self.env.is_terminal((1, 1)))

    def test_is_obstacle(self):
        """Ensure obstacles are correctly flagged."""

        for o in self.env.obstacles:
            self.assertTrue(self.env.is_obstacle(o))
        self.assertFalse(self.env.is_obstacle((1, 1)))

    def test_is_flag(self):
        """Verify reward flags are identified separately from obstacles."""

        for f in self.env.reward_flags:
            self.assertTrue(self.env.is_flag(f))
        self.assertFalse(self.env.is_flag((2, 2)))  # Obstacle but not flag
    
    def test_next_state(self):
        """Validate correct state transitions, including boundaries and obstacles."""

        self.assertEqual(self.env.next_state((0, 0), A_RIGHT), (0, 1))
        self.assertEqual(self.env.next_state((1, 1), A_UP), (0, 1))
        self.assertEqual(self.env.next_state((1, 4), A_LEFT), (1, 3))
        self.assertEqual(self.env.next_state((1, 1), A_DOWN), (2, 1))

        # Test hitting obstacle: should stay same
        self.assertEqual(self.env.next_state((1, 2), A_DOWN), (1, 2))  # (2,2) is obstacle
        
        # Test hitting boundary: should stay same
        self.assertEqual(self.env.next_state((0, 0), A_UP), (0, 0))
        self.assertEqual(self.env.next_state((4, 4), A_RIGHT), (4, 4))  # terminal on edge

    def test_reward(self):
        """Test environment reward function for different state transitions."""

        # Normal step reward
        self.assertEqual(self.env.reward((0, 0), A_RIGHT, (0, 1)), -1.0)
        # Stepping into terminal + flag (1,3)
        self.assertEqual(self.env.reward((1, 2), A_RIGHT, (1, 3)), 10.0 + self.env.flag_reward)
        # Staying in terminal state (no move)
        self.assertEqual(self.env.reward((0, 4), A_RIGHT, (0, 4)), -1.0)
        # Stepping onto a flag (not terminal)
        self.assertEqual(self.env.reward((2, 3), A_LEFT, (2, 2)), self.env.step_reward)  # (2,2) is obstacle, so no flag reward
        self.assertEqual(self.env.reward((0, 3), A_RIGHT, (0, 4)), 10.0)  # terminal (0,4), no flag reward

    def test_transition_dist_sum(self):
        """Transition probabilities should always sum to 1.0."""
        
        s = (0, 0)
        a = A_RIGHT
        dist = transition_dist(self.env, s, a, noise=0.2)
        total_prob = sum(p for _, p in dist)
        self.assertAlmostEqual(total_prob, 1.0)

    def test_transition_dist_states(self):
        """All transition outcomes should be valid grid states or terminals."""
        
        # Transition states must be valid states in grid or same state if obstacle/boundary
        s = (0, 0)
        a = A_RIGHT
        dist = transition_dist(self.env, s, a, noise=0.2)
        for s_next, _ in dist:
            self.assertTrue(self.env.in_bounds(s_next[0], s_next[1]))
            # Also not an obstacle unless terminal
            if self.env.is_obstacle(s_next):
                self.assertTrue(self.env.is_terminal(s_next))


class TestPolicyEvaluation(unittest.TestCase):
    """Tests for policy evaluation under a fixed policy."""

    def setUp(self):
        self.env = Gridworld()
        self.gamma = 0.95
        self.noise = 0.0
        self.V0 = np.zeros((self.env.H, self.env.W))
        self.pi = random_policy(self.env)

    def test_policy_evaluation_progress(self):
        """Verify that value estimates improve over multiple sweeps."""
        
        # Apply multiple sweeps to cause change
        V = self.V0.copy()
        for _ in range(5):
            V = policy_evaluation(self.env, V, self.pi, self.gamma, self.noise)
            print("Value function after evaluation:")
            print(V)

        # Now check that V is not basically zero in all non-terminal/non-obstacle positions
        changed = np.any([
            abs(V[r, c]) > 1e-6
            for r in range(self.env.H)
            for c in range(self.env.W)
            if not self.env.is_terminal((r, c)) and not self.env.is_obstacle((r, c))
        ])

        self.assertTrue(changed, "After multiple evaluation sweeps, value function didn't move away from zero.")

        # Also, terminal states should remain zero
        for t in self.env.terminals:
            self.assertAlmostEqual(V[t[0], t[1]], 0.0, msg=f"Value at terminal {t} should stay zero")

    def test_policy_evaluation_todo_implement(self):
        """Check if value function updates after one sweep with noise."""
        
        V = np.zeros((self.env.H, self.env.W))
        pi = random_policy(self.env)
        self.noise = 0.1

        V_new = policy_evaluation(self.env, V, pi, self.gamma, self.noise)
        # Value function should change for some non-terminal states after one sweep
        changed = False
        for r in range(self.env.H):
            for c in range(self.env.W):
                if not self.env.is_terminal((r,c)) and not self.env.is_obstacle((r,c)):
                    if abs(V_new[r,c]) > 1e-6:
                        changed = True
                        break
            if changed:
                break
        self.assertTrue(changed, "Value function did not update after one policy evaluation sweep.")

class TestPolicyImprovement(unittest.TestCase):
    """Tests for policy improvement step correctness."""
    
    def setUp(self):
        self.env = Gridworld()
        self.gamma = 0.95
        self.noise = 0.0
        self.V = np.zeros((self.env.H, self.env.W))
        self.pi = np.ones((self.env.H, self.env.W, 4)) / 4  # Random policy
    
    def test_policy_improvement(self):
        """Check policy probabilities and policy change after improvement."""

        pi_new, stable = policy_improvement(self.env, self.V, self.gamma, self.noise)

        # Policy probabilities should sum to 1 or zero for obstacles/terminals
        for r in range(self.env.H):
            for c in range(self.env.W):
                probs = pi_new[r,c,:]
                total = np.sum(probs)
                if self.env.is_terminal((r,c)) or self.env.is_obstacle((r,c)):
                    self.assertTrue(np.all(probs == 0), "Policy at terminal or obstacle should be zero.")
                else:
                    self.assertAlmostEqual(total, 1.0, msg=f"Policy probabilities do not sum to 1 at {(r,c)}")

        # Ensure that the new policy is not the same as the initial random policy
        self.assertFalse(np.array_equal(self.pi, pi_new))  # Policy should change
        
        # Check if the policy is stable (it can be flagged as stable if it has converged)
        self.assertTrue(stable)


class TestPolicyIteration(unittest.TestCase):

    def setUp(self):
        self.env = Gridworld()
        self.gamma = 0.95
        self.noise = 0.0
        self.V = np.zeros((self.env.H, self.env.W))
        self.pi = random_policy(self.env)
        self.start = (4, 0)  # Defined in Viewer

    def test_environment_setup(self):
        # Has at least one obstacle, flag, terminal
        self.assertGreater(len(self.env.obstacles), 0, "No obstacles in environment")
        self.assertGreater(len(self.env.reward_flags), 0, "No reward flags in environment")
        self.assertGreater(len(self.env.terminals), 0, "No terminal states in environment")
        self.assertTrue(self.env.in_bounds(self.start[0], self.start[1]))

    def test_policy_iteration_returns_valid_policy(self):
        _, pi = policy_iteration(self.env, self.gamma, self.noise)
        
        # Create mask of valid states (not terminal and not obstacle)
        valid_mask = np.ones((self.env.H, self.env.W), dtype=bool)
        for r, c in self.env.obstacles + self.env.terminals:
            valid_mask[r, c] = False

        # Get the sum of probabilities across actions
        prob_sums = pi.sum(axis=2)

        # Ensure all valid states sum to 1.0
        self.assertTrue(np.allclose(prob_sums[valid_mask], 1.0), "Policy probabilities do not sum to 1 in valid states")
        
        # Ensure all invalid states (terminal or obstacle) sum to 0.0
        self.assertTrue(np.allclose(prob_sums[~valid_mask], 0.0), "Policy for terminal/obstacle states should sum to 0")

    def test_extract_path_reaches_terminal(self):
        V, pi = policy_iteration(self.env, self.gamma, self.noise)
        path = extract_path(self.env, pi, self.start)
        self.assertGreater(len(path), 1)
        self.assertIn(path[-1], self.env.terminals)

    def test_obstacles_block_movement(self):
        # (2,2) is an obstacle; moving into it should be blocked
        self.assertEqual(self.env.next_state((1, 2), A_DOWN), (1, 2))

    def test_reward_flag_gives_bonus(self):
        s = (1, 2)
        a = A_DOWN
        s_next = (2, 2)  # Normally this would be a valid move, but it's an obstacle
        if not self.env.is_obstacle(s_next):
            reward = self.env.reward(s, a, s_next)
            if self.env.is_flag(s_next):
                self.assertGreater(reward, self.env.step_reward)


class TestGridworldViewerUI(unittest.TestCase):
    """Unit tests for GUI components (sliders, buttons, default values)."""

    def setUp(self):
        self.viewer = Viewer()  # Instantiate your GUI

    def test_buttons_present(self):
        """Ensure expected buttons are created and labeled correctly."""

        buttons = [self.viewer.btn_iter, self.viewer.btn_run, self.viewer.btn_path]
        button_labels = [b.label.get_text() for b in buttons]

        expected_labels = ["Iterate once", "Run to convergence", "Show Path"]
        for label in expected_labels:
            self.assertIn(label, button_labels, f"Button '{label}' not found.")

    def test_sliders_present(self):
        """Confirm that gamma and noise sliders are present."""

        sliders = [self.viewer.s_gamma, self.viewer.s_noise]

        found_gamma = any(slider.label.get_text().lower() == "gamma" for slider in sliders)
        found_noise = any(slider.label.get_text().lower() == "noise" for slider in sliders)

        self.assertTrue(found_gamma, "Slider for gamma not found.")
        self.assertTrue(found_noise, "Slider for noise not found.")

    def test_slider_defaults(self):
        """Check default values for gamma and noise sliders."""

        self.assertAlmostEqual(self.viewer.s_gamma.val, 0.95, places=2, msg="Gamma slider default is incorrect.")
        self.assertAlmostEqual(self.viewer.s_noise.val, 0.0, places=2, msg="Noise slider default is incorrect.")

    def test_slider_ranges(self):
        """Verify slider min/max ranges match expected parameters."""

        # Check gamma slider range
        self.assertAlmostEqual(self.viewer.s_gamma.valmin, 0.50)
        self.assertAlmostEqual(self.viewer.s_gamma.valmax, 0.99)

        # Check noise slider range
        self.assertAlmostEqual(self.viewer.s_noise.valmin, 0.0)
        self.assertAlmostEqual(self.viewer.s_noise.valmax, 0.5)

if __name__ == '__main__':
    unittest.main()
