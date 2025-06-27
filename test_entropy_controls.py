#!/usr/bin/env python3
"""
Test script for temporal clustering and prompt entropy controls.
Verifies the enhanced statistical protocols work as expected.
"""

import sys
import time
import json
from batch_dream_generator import (
    SamplingConfig, 
    PromptEntropyGenerator, 
    TemporalDispersionManager
)

def test_prompt_entropy_generator():
    """Test the prompt entropy generator functionality."""
    print("=== Testing Prompt Entropy Generator ===")
    
    generator = PromptEntropyGenerator(variant_types=5)
    base_prompt = "Finish: Last night I dreamt of…"
    
    print(f"Base prompt: '{base_prompt}'")
    print("\nTesting marker generation:")
    
    results = []
    for i in range(10):
        modified_prompt, prompt_id, marker_info = generator.generate_prompt_variant(
            base_prompt, use_markers=(i % 2 == 0)
        )
        results.append({
            'prompt_id': prompt_id,
            'marker_info': marker_info,
            'modified_prompt': modified_prompt,
            'has_markers': (i % 2 == 0)
        })
        print(f"  {i+1}: ID={prompt_id}, Marker={marker_info}, Modified='{modified_prompt}'")
    
    # Verify all prompt IDs are unique
    prompt_ids = [r['prompt_id'] for r in results]
    assert len(set(prompt_ids)) == len(prompt_ids), "Prompt IDs should be unique"
    
    # Verify marker usage
    with_markers = [r for r in results if r['has_markers']]
    without_markers = [r for r in results if not r['has_markers']]
    
    print(f"\nResults: {len(with_markers)} with markers, {len(without_markers)} without")
    
    # Check that markers were actually applied when requested
    markers_applied = [r for r in with_markers if r['marker_info'] != 'none']
    print(f"Markers successfully applied: {len(markers_applied)}/{len(with_markers)}")
    
    print("✓ Prompt Entropy Generator tests passed\n")
    return True

def test_temporal_dispersion_manager():
    """Test the temporal dispersion manager."""
    print("=== Testing Temporal Dispersion Manager ===")
    
    config = SamplingConfig(
        min_temporal_dispersion_minutes=1,  # 1 minute for testing
        max_temporal_dispersion_hours=2,     # 2 hours for testing
        temporal_dispersion_hours=1
    )
    
    manager = TemporalDispersionManager(config)
    
    print(f"Config: min={config.min_temporal_dispersion_minutes}min, max={config.max_temporal_dispersion_hours}h")
    
    # Test delay calculation
    delays = []
    extended_delays = 0
    
    print("\nTesting delay calculations:")
    for i in range(20):
        delay = manager.get_next_call_delay()
        delays.append(delay)
        
        # Record call time for statistics
        manager.record_call_time()
        time.sleep(0.01)  # Small sleep to create time differences
        
        is_extended = delay > (config.min_temporal_dispersion_minutes * 60 * 3)
        if is_extended:
            extended_delays += 1
        
        # Skip the first call (which returns 0) in output for clarity
        if i == 0 and delay == 0:
            print(f"  {i+1}: {delay:.1f}s (first call - no delay expected)")
        else:
            print(f"  {i+1}: {delay:.1f}s ({delay/60:.1f}min) {'[EXTENDED]' if is_extended else ''}")
    
    # Verify delay ranges (excluding first call which is always 0)
    min_expected = config.min_temporal_dispersion_minutes * 60
    max_basic = min_expected * 3
    
    # Filter out the first delay (which is always 0) for analysis
    analysis_delays = delays[1:] if delays and delays[0] == 0 else delays
    
    basic_delays = [d for d in analysis_delays if d <= max_basic]
    extended_delay_list = [d for d in analysis_delays if d > max_basic]
    
    print(f"\nDelay analysis (excluding first call):")
    print(f"  Total analyzed delays: {len(analysis_delays)}")
    print(f"  Basic delays: {len(basic_delays)} (range: {min(basic_delays):.1f}s - {max(basic_delays):.1f}s)" if basic_delays else "  Basic delays: 0")
    print(f"  Extended delays: {len(extended_delay_list)} (expected ~10% = {len(analysis_delays)*0.1:.1f})")
    print(f"  Extended delay rate: {len(extended_delay_list)/len(analysis_delays)*100:.1f}%" if analysis_delays else "  Extended delay rate: 0%")
    
    # Test temporal statistics
    stats = manager.get_temporal_statistics()
    print(f"\nTemporal statistics:")
    print(f"  Intervals recorded: {len(stats['intervals'])}")
    print(f"  Mean interval: {stats['mean_interval']:.3f}s")
    print(f"  Std interval: {stats['std_interval']:.3f}s")
    print(f"  Total span: {stats['total_span_hours']:.6f}h")
    
    # Verify basic constraints (only for non-zero delays)
    if basic_delays:
        assert all(d >= min_expected * 0.99 for d in basic_delays), "Basic delays should be >= minimum"
    assert len(extended_delay_list) <= len(analysis_delays) * 0.3, "Extended delays should be reasonable"
    
    print("✓ Temporal Dispersion Manager tests passed\n")
    return True

def test_sampling_config():
    """Test the enhanced sampling configuration."""
    print("=== Testing Enhanced Sampling Config ===")
    
    # Test default configuration
    config = SamplingConfig()
    print("Default configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Temporal dispersion: {config.temporal_dispersion_hours}h")
    print(f"  Min temporal dispersion: {config.min_temporal_dispersion_minutes}min")
    print(f"  Max temporal dispersion: {config.max_temporal_dispersion_hours}h")
    print(f"  Use prompt variants: {config.use_prompt_variants}")
    print(f"  Marker probability: {config.invisible_marker_probability}")
    print(f"  Variant types: {config.prompt_variant_types}")
    
    # Test custom configuration
    custom_config = SamplingConfig(
        min_temporal_dispersion_minutes=45,
        max_temporal_dispersion_hours=48,
        invisible_marker_probability=0.5,
        use_prompt_variants=False
    )
    
    print("\nCustom configuration:")
    print(f"  Min temporal dispersion: {custom_config.min_temporal_dispersion_minutes}min")
    print(f"  Max temporal dispersion: {custom_config.max_temporal_dispersion_hours}h")
    print(f"  Use prompt variants: {custom_config.use_prompt_variants}")
    print(f"  Marker probability: {custom_config.invisible_marker_probability}")
    
    # Verify types and ranges
    assert isinstance(config.use_prompt_variants, bool), "use_prompt_variants should be boolean"
    assert 0.0 <= config.invisible_marker_probability <= 1.0, "Marker probability should be 0-1"
    assert config.min_temporal_dispersion_minutes > 0, "Min dispersion should be positive"
    assert config.max_temporal_dispersion_hours > config.temporal_dispersion_hours, "Max should be > regular dispersion"
    
    print("✓ Enhanced Sampling Config tests passed\n")
    return True

def test_integration():
    """Test integration of all components."""
    print("=== Testing Integration ===")
    
    config = SamplingConfig(
        min_temporal_dispersion_minutes=1,
        use_prompt_variants=True,
        invisible_marker_probability=0.6
    )
    
    entropy_gen = PromptEntropyGenerator(config.prompt_variant_types)
    temporal_mgr = TemporalDispersionManager(config)
    
    print("Running integrated test simulation...")
    
    base_prompts = [
        "Finish: Last night I dreamt of…",
        "Amaitu: Bart amets egin nuen…",
        "Završi: Sinoć sam sanjao…"
    ]
    
    results = []
    for i, prompt in enumerate(base_prompts):
        # Generate entropy variant
        use_markers = config.use_prompt_variants and (
            __import__('random').random() < config.invisible_marker_probability
        )
        
        modified_prompt, prompt_id, marker_info = entropy_gen.generate_prompt_variant(
            prompt, use_markers
        )
        
        # Get temporal delay
        delay = temporal_mgr.get_next_call_delay()
        temporal_mgr.record_call_time()
        
        result = {
            'call_id': f"test_{i}",
            'base_prompt': prompt,
            'modified_prompt': modified_prompt,
            'prompt_id': prompt_id,
            'marker_info': marker_info,
            'used_invisible_markers': use_markers,
            'temporal_delay': delay
        }
        results.append(result)
        
        print(f"  Call {i+1}: ID={prompt_id}, Markers={use_markers}, Delay={delay:.1f}s")
    
    # Analyze results
    with_markers = [r for r in results if r['used_invisible_markers']]
    unique_ids = set(r['prompt_id'] for r in results)
    
    print(f"\nIntegration results:")
    print(f"  Total calls: {len(results)}")
    print(f"  Calls with markers: {len(with_markers)}")
    print(f"  Marker usage rate: {len(with_markers)/len(results)*100:.1f}%")
    print(f"  Unique prompt IDs: {len(unique_ids)}")
    
    # Export sample result
    sample_output = {
        'test_session': 'entropy_controls_test',
        'timestamp': time.time(),
        'config': {
            'min_temporal_dispersion_minutes': config.min_temporal_dispersion_minutes,
            'use_prompt_variants': config.use_prompt_variants,
            'invisible_marker_probability': config.invisible_marker_probability
        },
        'results': results,
        'temporal_statistics': temporal_mgr.get_temporal_statistics()
    }
    
    with open('test_entropy_output.json', 'w', encoding='utf-8') as f:
        json.dump(sample_output, f, ensure_ascii=False, indent=2)
    
    print(f"  Sample output exported to: test_entropy_output.json")
    
    print("✓ Integration tests passed\n")
    return True

def main():
    """Run all tests."""
    print("Running Temporal Clustering and Prompt Entropy Tests\n")
    
    try:
        # Run all test components
        tests = [
            test_sampling_config,
            test_prompt_entropy_generator,
            test_temporal_dispersion_manager,
            test_integration
        ]
        
        passed = 0
        for test_func in tests:
            if test_func():
                passed += 1
        
        print(f"=== Test Summary ===")
        print(f"Tests passed: {passed}/{len(tests)}")
        
        if passed == len(tests):
            print("✓ All temporal clustering and prompt entropy controls working correctly!")
            return 0
        else:
            print("✗ Some tests failed")
            return 1
            
    except Exception as e:
        print(f"✗ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 