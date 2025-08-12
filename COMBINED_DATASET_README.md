# TinyML-Autopilot Combined Benchmark Dataset

## Overview
This document describes the combined CSV dataset containing all benchmark results from the TinyML-Autopilot project testing across 2025.

**File**: `combined_tinyml_benchmark_data.csv`
**Total Records**: 1,780 rows
**Unique Batch IDs**: 54
**Time Period**: January 15, 2025 - August 7, 2025

## Dataset Structure

### Core Columns (from original CSV files)
- `num_run`: Sequential run number within each batch
- `name`: Unique identifier for each test run
- `trace_id`: Unique trace identifier for monitoring
- `batch_id`: Batch identifier (key categorization column)
- `status`: Test result status (success/failure)
- `latency`: API response latency in seconds
- `total_tokens`: Total tokens used in LLM interaction
- `prompt_tokens`: Input tokens to LLM
- `completion_tokens`: Output tokens from LLM
- `total_cost`: API cost (currently 0.0 for local models)
- `parameters`: LLM parameters used (temperature, top_p)
- `prompt_cost`: Cost for input tokens
- `completion_cost`: Cost for output tokens
- `tags`: Test metadata tags (model, processor type, benchmark flag)
- `timestamp`: Unix timestamp of test execution

### Added Metadata Columns
- `source_file`: Original CSV filename
- `source_path`: Full path to original CSV file
- `test_date`: Extracted test date (e.g., "08.07")
- `model_config`: Model configuration identifier (e.g., "phi4_3b43")

## Test Categories by Processor Type

### PySketchGenerator (PSG) - Python TensorFlow Lite Scripts
- **Purpose**: Generate Python scripts for TensorFlow Lite inference on edge devices
- **Batch ID Pattern**: `*_psg_batch`
- **Target Platforms**: Raspberry Pi, general edge devices

### TPUSketchGenerator (TPUSG) - TPU-Optimized Scripts  
- **Purpose**: Generate TPU-optimized Python scripts for Google Coral devices
- **Batch ID Pattern**: `*_tpusg_batch`
- **Target Platforms**: Google Coral Edge TPU devices

### SketchGenerator (SG) - Arduino C++ Sketches
- **Purpose**: Generate Arduino C++ sketches for microcontrollers
- **Batch ID Pattern**: `*_sg_batch`
- **Target Platforms**: Arduino Nano 33 BLE Sense, other microcontrollers

### DataProcessor (DP) - Data Pipeline Generation
- **Purpose**: Generate data preprocessing pipelines
- **Batch ID Pattern**: `*_dp_batch`
- **Focus**: Data preprocessing workflow automation

### ModelConverter (MC) - Model Conversion Scripts
- **Purpose**: Generate TensorFlow to TensorFlow Lite conversion scripts
- **Batch ID Pattern**: `*_mc_batch`
- **Focus**: Model optimization and conversion

## Model Coverage

### Large Language Models Tested
1. **Qwen2.5-Coder** (14B and 32B parameters)
   - Most extensively tested model
   - Variants: qwen2.5-coder:14b, qwen2.5-coder:32b
   
2. **Phi-4** (Microsoft)
   - Multiple configurations tested
   - Strong performance in code generation
   
3. **Codestral** (Mistral AI)
   - Specialized code generation model
   - Multiple test configurations
   
4. **DeepSeek-R1** (14B parameters)
   - Reasoning-focused model
   - Limited testing compared to others
   
5. **Llama 3.1** (Various sizes)
   - Meta's open-source model
   - Multiple configuration tests
   
6. **Gemma 3** (27B parameters)
   - Google's model
   - Limited testing period

## Test Distribution by Date

### High-Activity Periods
- **March 18, 2025**: 443 tests (largest single-day testing)
- **January 15-23, 2025**: 240 tests (initial benchmark establishment)
- **March 22-28, 2025**: 257 tests (intensive model comparison)

### Recent Testing (July-August 2025)
- **July 28-August 7**: 480 tests
- Focus on PSG and TPUSG processors
- Primarily Qwen2.5-Coder and Phi-4 models

## Key Performance Metrics

### Success Rate Analysis
- Track generation success across different models and tasks
- Identify failure patterns and common error modes
- Compare processor-specific performance

### Latency Profiling
- API response times ranging from ~13s (success) to ~120s (failure)
- Network latency measurements for optimization
- Model-specific performance characteristics

### Token Usage Patterns
- Successful runs: ~2,000-3,000 tokens
- Failed runs: ~13,000+ tokens (multiple retry attempts)
- Cost implications for different models

## Research Applications

### Model Comparison Studies
- Direct A/B testing between different LLMs
- Performance regression tracking over time
- Task-specific model optimization

### Code Generation Quality Assessment
- Compilation success rates for different target platforms
- Runtime validation success across edge devices
- Hardware-specific optimization effectiveness

### Development Workflow Optimization
- Identify bottlenecks in automated code generation
- Optimize prompt engineering strategies
- Improve error recovery mechanisms

## Data Quality Notes

### Batch Consistency
- Most batches contain 30 runs for statistical significance
- Some incomplete batches (17-23 runs) due to testing interruptions
- Consistent metadata tracking across all tests

### Temporal Coverage
- 7-month testing period provides longitudinal insights
- Multiple model versions and configurations tested
- Evolution of framework capabilities over time

## Usage Recommendations

### Statistical Analysis
- Use batch_id for grouping related tests
- Consider test_date for temporal analysis
- Filter by processor type for task-specific studies

### Performance Benchmarking
- Compare latency across models and tasks
- Analyze token efficiency patterns
- Track success rates by configuration

### Research Applications
- Longitudinal studies of LLM code generation capabilities
- Cross-model performance comparison
- Edge AI deployment automation effectiveness

---

**Generated**: August 10, 2025
**Dataset Version**: Combined from 61 individual CSV files
**Framework**: TinyML-Autopilot Benchmark Suite
