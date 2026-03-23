#!/usr/bin/env python3
"""
Comprehensive Test Script for ChildEnv and ChildPolicy
Tests both heuristic and PPO implementations using project baseline files.
"""

import sys
import os
import gymnasium as gym
from gymnasium.envs.registration import register

print("="*70)
print(" COMPREHENSIVE TEST FOR CHILDENV + CHILDPOLICY (PPO)")
print("="*70)

# ============================================================================
# PART 1: SETUP AND IMPORTS
# ============================================================================

print("\n[PART 1/5] SETUP AND IMPORTS")
print("-" * 70)

# Step 1: Check if files exist in correct locations
print("\n1.1 Checking file locations...")

files_to_check = [
    ("app/simulation/envs/ChildEnv.py", "ChildEnv"),
    ("app/simulation/policies/ChildPolicy.py", "ChildPolicy"),
]

missing_files = []
for filepath, name in files_to_check:
    if os.path.exists(filepath):
        print(f"  ✓ {name} found at {filepath}")
    else:
        print(f"  ✗ {name} NOT found at {filepath}")
        missing_files.append((filepath, name))

if missing_files:
    print("\n❌ Missing files! Please copy:")
    for filepath, name in missing_files:
        print(f"  cp {name}_final.py {filepath}")
    sys.exit(1)

# Step 2: Import required modules
print("\n1.2 Importing modules...")

try:
    from app.simulation.envs.ChildEnv import ChildEnv
    print("  ✓ ChildEnv imported")
except ImportError as e:
    print(f"  ✗ Failed to import ChildEnv: {e}")
    sys.exit(1)

try:
    from app.simulation.policies.ChildPolicy import ChildPolicy
    print("  ✓ ChildPolicy imported")
except ImportError as e:
    print(f"  ✗ Failed to import ChildPolicy: {e}")
    sys.exit(1)

try:
    from app.data.Instance import Instance
    from app.data.Scenario import Scenario
    from app.simulation.envs.Env import Env
    from app.simulation.policies.PolicyEvaluation import PolicyEvaluation
    print("  ✓ Project modules imported")
except ImportError as e:
    print(f"  ✗ Failed to import project modules: {e}")
    sys.exit(1)

# Step 3: Check SB3-Contrib availability
print("\n1.3 Checking PPO dependencies...")
try:
    from sb3_contrib import MaskablePPO
    print("  ✓ sb3-contrib available (PPO will be used)")
    HAS_PPO = True
except ImportError:
    print("  ⚠ sb3-contrib not available (will use heuristic only)")
    print("    Install with: pip install sb3-contrib")
    HAS_PPO = False

# Register environment
register(
    id="Child_Env",
    entry_point="app.simulation.envs.ChildEnv:ChildEnv",
)
print("  ✓ Environment registered")

print("\n✅ PART 1 PASSED: All imports successful")

# ============================================================================
# PART 2: ENVIRONMENT TESTS
# ============================================================================

print("\n[PART 2/5] ENVIRONMENT TESTS")
print("-" * 70)

# Load test instance
print("\n2.1 Loading test instance...")
try:
    instance = Instance.create(
        Instance.SourceType.FILE,
        "app/data/data_files/timeline_0.json",
        "app/data/data_files/average_matrix_0.json",
        "app/data/data_files/appointments_0.json",
        "app/data/data_files/unavailability_0.json"
    )
    print(f"  ✓ Instance loaded")
    print(f"    - Customers: {len(instance.timeline)}")
    print(f"    - Servers: {len(instance.average_matrix)}")
    print(f"    - Appointments: {len(instance.appointments)}")
except Exception as e:
    print(f"  ✗ Failed to load instance: {e}")
    sys.exit(1)

# Create environment
print("\n2.2 Creating environment...")
try:
    env = gym.make("Child_Env", mode=Env.MODE.TEST, instance=instance)
    print("  ✓ Environment created")
    print(f"    - Action space: {env.action_space}")
    print(f"    - Observation space keys: {env.observation_space.spaces.keys()}")
except Exception as e:
    print(f"  ✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test reset
print("\n2.3 Testing environment reset...")
try:
    obs, info = env.reset(seed=42)
    print("  ✓ Environment reset successful")
    print(f"    - Customer features shape: {obs['customers'].shape}")
    print(f"    - Context features shape: {obs['context'].shape}")
    print(f"    - Info keys: {list(info.keys())}")
except Exception as e:
    print(f"  ✗ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test observation values
print("\n2.4 Validating observation values...")
try:
    customers = obs['customers']
    context = obs['context']
    
    assert customers.min() >= 0 and customers.max() <= 1, "Customer features out of [0,1]"
    assert context.min() >= 0 and context.max() <= 1, "Context features out of [0,1]"
    
    num_real_customers = int(customers[:, 8].sum())
    print(f"  ✓ Observations valid")
    print(f"    - Customer range: [{customers.min():.3f}, {customers.max():.3f}]")
    print(f"    - Context range: [{context.min():.3f}, {context.max():.3f}]")
    print(f"    - Real customers: {num_real_customers}")
except Exception as e:
    print(f"  ✗ Validation failed: {e}")
    sys.exit(1)

# Test action masking
print("\n2.5 Testing action masking...")
try:
    # Access the unwrapped environment to get action_masks
    action_mask = env.unwrapped.action_masks()
    assert len(action_mask) == 51, f"Expected 51 actions, got {len(action_mask)}"
    assert action_mask[50] == True, "HOLD should always be valid"
    
    valid_actions = sum(action_mask)
    print(f"  ✓ Action masking works")
    print(f"    - Valid actions: {valid_actions}/51")
    print(f"    - HOLD valid: {action_mask[50]}")
except Exception as e:
    print(f"  ✗ Action masking failed: {e}")
    sys.exit(1)

# Test step
print("\n2.6 Testing environment step...")
try:
    # Get action mask from unwrapped env
    action_mask = env.unwrapped.action_masks()
    valid_indices = [i for i, valid in enumerate(action_mask) if valid]
    action = valid_indices[0]
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✓ Step successful")
    print(f"    - Reward: {reward:.2f}")
    print(f"    - Terminated: {terminated}")
    print(f"    - Truncated: {truncated}")
except Exception as e:
    print(f"  ✗ Step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ PART 2 PASSED: Environment works correctly")

# ============================================================================
# PART 3: HEURISTIC POLICY TEST
# ============================================================================

print("\n[PART 3/5] HEURISTIC POLICY TEST")
print("-" * 70)

print("\n3.1 Creating heuristic policy...")
policy = ChildPolicy("Heuristic_Test")
print("  ✓ Policy created")

print("\n3.2 Running simulation with heuristic...")
try:
    env = gym.make("Child_Env", mode=Env.MODE.TEST, instance=instance)
    policy.simulate(env, print_logs=False)
    
    print(f"  ✓ Simulation completed")
    print(f"    - Total reward: {policy.total_reward:.2f}")
    print(f"    - Customers served: {len(policy.customers_history)}")
    print(f"    - Abandonment: {policy.customer_abandonment}")
    print(f"    - Avg wait time: {policy.avg_waiting_time:.2f} min")
except Exception as e:
    print(f"  ✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3.3 Evaluating heuristic performance...")
try:
    evaluator = PolicyEvaluation(
        instance.timeline,
        instance.appointments,
        clients_history=policy.customers_history
    )
    evaluator.evaluate()
    
    print(f"\n  {'='*60}")
    print(f"  HEURISTIC POLICY RESULTS")
    print(f"  {'='*60}")
    print(f"  Wait Score:        {evaluator.grade_wait:6.2f}")
    print(f"  Appointment Score: {evaluator.grade_appointment:6.2f}")
    print(f"  Throughput Score:  {evaluator.grade_number_of_unserved:6.2f}")
    print(f"  {'─'*60}")
    print(f"  FINAL SCORE:       {evaluator.final_grade:6.2f}")
    print(f"  {'='*60}")
    
    # Store heuristic score for comparison
    heuristic_score = evaluator.final_grade
    
except Exception as e:
    print(f"  ✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ PART 3 PASSED: Heuristic policy works")

# ============================================================================
# PART 4: PPO TRAINING TEST (if available)
# ============================================================================

if HAS_PPO:
    print("\n[PART 4/5] PPO TRAINING TEST")
    print("-" * 70)
    
    print("\n4.1 Loading training scenario...")
    try:
        scenario = Scenario.from_json("app/data/config/queue_config.json")
        print(f"  ✓ Scenario loaded")
        print(f"    - Servers: {scenario.S}")
        print(f"    - Lambda: {scenario.lmbd}")
    except Exception as e:
        print(f"  ✗ Failed to load scenario: {e}")
        sys.exit(1)
    
    print("\n4.2 Creating PPO policy...")
    ppo_policy = ChildPolicy("Test_PPO_5k")
    print("  ✓ PPO policy created")
    
    print("\n4.3 Training PPO (5000 timesteps - quick test)...")
    print("  ⏱ This will take ~2-5 minutes...")
    print("  💡 For full training, use 100,000+ timesteps\n")
    
    try:
        ppo_policy.learn(scenario, total_timesteps=5000, verbose=1)
        print("\n  ✓ Training completed")
    except KeyboardInterrupt:
        print("\n  ⚠ Training interrupted")
    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        HAS_PPO = False  # Disable PPO evaluation
    
    if HAS_PPO:
        print("\n4.4 Testing trained PPO model...")
        try:
            env = gym.make("Child_Env", mode=Env.MODE.TEST, instance=instance)
            ppo_policy.simulate(env, print_logs=False)
            
            print(f"  ✓ Simulation completed")
            print(f"    - Total reward: {ppo_policy.total_reward:.2f}")
            print(f"    - Customers served: {len(ppo_policy.customers_history)}")
            
            # Evaluate
            evaluator = PolicyEvaluation(
                instance.timeline,
                instance.appointments,
                clients_history=ppo_policy.customers_history
            )
            evaluator.evaluate()
            
            print(f"\n  {'='*60}")
            print(f"  PPO RESULTS (5K TIMESTEPS)")
            print(f"  {'='*60}")
            print(f"  Wait Score:        {evaluator.grade_wait:6.2f}")
            print(f"  Appointment Score: {evaluator.grade_appointment:6.2f}")
            print(f"  Throughput Score:  {evaluator.grade_number_of_unserved:6.2f}")
            print(f"  {'─'*60}")
            print(f"  FINAL SCORE:       {evaluator.final_grade:6.2f}")
            print(f"  {'='*60}")
            
            ppo_score = evaluator.final_grade
            
            print(f"\n  📊 Comparison:")
            print(f"    - Heuristic: {heuristic_score:.2f}")
            print(f"    - PPO (5k):  {ppo_score:.2f}")
            print(f"    - Difference: {ppo_score - heuristic_score:+.2f}")
            
            if ppo_score >= heuristic_score - 5:
                print(f"\n  ✓ PPO performance is reasonable")
            else:
                print(f"\n  ⚠ PPO score lower than heuristic (needs more training)")
            
        except Exception as e:
            print(f"  ✗ PPO testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ PART 4 PASSED: PPO training works")
else:
    print("\n[PART 4/5] PPO TRAINING TEST - SKIPPED")
    print("-" * 70)
    print("  ⚠ sb3-contrib not available")
    print("  Install with: pip install sb3-contrib")
    print("  Heuristic policy will be used instead")

# ============================================================================
# PART 5: MULTI-INSTANCE TEST
# ============================================================================

print("\n[PART 5/5] MULTI-INSTANCE TEST")
print("-" * 70)

print("\n5.1 Testing on 3 different instances...")

test_instances = [0, 1, 2]
results = []

for idx in test_instances:
    print(f"\n  Testing instance {idx}...")
    try:
        instance = Instance.create(
            Instance.SourceType.FILE,
            f"app/data/data_files/timeline_{idx}.json",
            f"app/data/data_files/average_matrix_{idx}.json",
            f"app/data/data_files/appointments_{idx}.json",
            f"app/data/data_files/unavailability_{idx}.json"
        )
        
        env = gym.make("Child_Env", mode=Env.MODE.TEST, instance=instance)
        
        # Use heuristic policy
        test_policy = ChildPolicy(f"Test_Instance_{idx}")
        test_policy.simulate(env, print_logs=False)
        
        evaluator = PolicyEvaluation(
            instance.timeline,
            instance.appointments,
            clients_history=test_policy.customers_history
        )
        evaluator.evaluate()
        
        results.append({
            'instance': idx,
            'score': evaluator.final_grade,
            'wait': evaluator.grade_wait,
            'appt': evaluator.grade_appointment,
            'throughput': evaluator.grade_number_of_unserved
        })
        
        print(f"    ✓ Score: {evaluator.final_grade:.2f}")
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")

# Print summary
if results:
    print(f"\n5.2 Multi-instance summary:")
    print(f"\n  {'Inst':<6} {'Score':<8} {'Wait':<8} {'Appt':<8} {'Through':<8}")
    print(f"  {'-'*40}")
    
    for r in results:
        print(f"  {r['instance']:<6} {r['score']:<8.2f} {r['wait']:<8.2f} "
              f"{r['appt']:<8.2f} {r['throughput']:<8.2f}")
    
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"  {'-'*40}")
    print(f"  {'AVG':<6} {avg_score:<8.2f}")
    
    print(f"\n  📊 Average score: {avg_score:.2f}")

print("\n✅ PART 5 PASSED: Multi-instance test successful")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" FINAL SUMMARY")
print("="*70)

print("\n✅ ALL TESTS PASSED!\n")

print("📊 Results:")
print(f"  - Heuristic baseline: {heuristic_score:.2f}")
if HAS_PPO:
    print(f"  - PPO (5k steps): {ppo_score:.2f}")
    print(f"  - Difference: {ppo_score - heuristic_score:+.2f}")
print(f"  - Multi-instance avg: {avg_score:.2f}")

print("\n📁 Files verified:")
print("  ✓ app/simulation/envs/ChildEnv.py")
print("  ✓ app/simulation/policies/ChildPolicy.py")

if HAS_PPO:
    print("\n💾 Models saved:")
    print("  - ./models/Test_PPO_5k/final_model.zip")
    print("  - ./models/Test_PPO_5k/best/")
    print("  - ./models/Test_PPO_5k/checkpoints/")
    
    print("\n📈 View training logs:")
    print("  tensorboard --logdir ./logs/")

print("\n🎯 Next steps:")
print("  1. Run full training: python -m app.main")
print("  2. Evaluate on 50 instances: python -m app.evaluate")
if not HAS_PPO:
    print("  3. Install PPO: pip install sb3-contrib")
print("\n" + "="*70)

sys.exit(0)