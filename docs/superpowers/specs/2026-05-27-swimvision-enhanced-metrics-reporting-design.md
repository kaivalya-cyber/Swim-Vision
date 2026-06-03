# SwimVision Enhanced Metrics and Reporting Design
**Date**: 2026-05-27  
**Author**: Claude Code (Brainstorming Session)

## Overview
This design outlines enhancements to SwimVision focusing on two key areas:
1. **Enhanced Metrics Suite** - Additional biomechanical measurements beyond current joint angles
2. **Advanced Reporting & Visualization** - Improved ways to visualize, compare, and report results

These enhancements maintain backward compatibility while extending functionality for deeper analysis and better user experience.

## Enhanced Metrics Suite

### New Biomechanical Measurements
| Metric Category | Specific Metrics | Description | Value to Coach/Athlete |
|----------------|------------------|-------------|------------------------|
| **Kinematics** | Joint velocity profiles (angular/linear) | Rate of change of joint angles over time | Identify explosive power generation and timing |
|  | Joint acceleration profiles | Rate of change of joint velocity | Understand force application patterns |
| **Symmetry Analysis** | Left/right joint angle comparison | Phase-wise differences between left/right limbs | Detect imbalances that may indicate technique flaws or injury risk |
|  | Left/right velocity/acceleration comparison | Temporal symmetry of movement patterns | Identify timing asymmetries in limb movements |
| **Alignment Metrics** | Head alignment axis | Deviation of head position from body line during flight/entry | Assess streamline maintenance and head position |
|  | Hip rotation angle | Transverse plane rotation of pelvis during block and flight | Evaluate rotational torque generation |
|  | Shoulder/hip separation angle | Timing and magnitude of upper/lower body separation | Assess kinetic chain efficiency |
| **Entry Analysis** | Entry trajectory angle | Descent angle relative to water surface (first 5 frames post-entry) | Evaluate entry mechanics for minimal splash and drag |
|  | Underwater glide metrics (where visible) | Depth maintenance and lateral movement during initial glide | Assess streamline effectiveness post-entry |
| **Dynamic Estimates** | Center of mass velocity | Whole-body horizontal velocity derived from key point tracking | Measure overall propulsive efficiency |
|  | Relative joint contributions | Percentage contribution of each joint to total movement | Identify which joints drive performance |

### Implementation Approach
- **Modular Design**: New metrics implemented as independent calculation modules in `src/metrics/`
- **Standard Interface**: Each metric follows a consistent API: input keypoints/confidence, output time-series data
- **Configuration System**: Enable/disable metrics via configuration flags to control computational load
- **Units Standardization**: All metrics use consistent units (degrees for angles, deg/s for velocity, deg/s² for acceleration, meters for distances where applicable)
- **Missing Data Handling**: Propagate NaN values appropriately through calculation chains

## Advanced Reporting & Visualization System

### Web-Based Interactive Dashboard
| Feature | Description | Technical Implementation |
|---------|-------------|--------------------------|
| **Zoomable Timeline View** | Scrollable timeline showing all metrics over the swim start sequence | React-based timeline component with syncronized metric panels |
| **Toggleable Overlay Layers** | User-selectable visualization of skeleton, joint vectors, metric values on video | HTML5 Canvas overlay with WebGL acceleration option |
| **Multi-Metric Comparison** | Ability to plot multiple metrics on shared time axis for correlation analysis | D3.js or Chart.js for synchronized multi-axis charts |
| **Customizable Layouts** | Drag-and-drop panels to create personalized analysis views | React Grid Layout or similar library |
| **Export Functionality** | Save current view as image, export metric data as CSV/JSON | Client-side canvas-to-blob and data serialization |

### Enhanced Reporting Capabilities
| Feature | Description | Benefit |
|---------|-------------|---------|
| **Side-by-Side Trial Comparison** | Synchronized playback and metric comparison of 2+ trials | Direct visualization of technique changes over time |
| **Progress Tracking Dashboard** | Longitudinal view of key metrics across multiple sessions | Objective measurement of training effectiveness |
| **Template-Based Reports** | Customizable report structures (coach summary, athlete feedback, research export) | Adapt output to different audience needs |
| **Statistical Summary Overlays** | Mean/median/std dev bands shown on time-series plots | Quick assessment of consistency and variability |
| **Annotation Tools** | Ability to add text, arrows, and highlights to video frames | Communication of specific coaching points |

### Reporting Architecture
- **Template Engine**: Jinja2 or similar for generating HTML/PDF reports from data templates
- **Chart Generation Library**: Pre-rendered static charts for PDF reports (Matplotlib/Seaborn)
- **Interactive Widgets**: React components for web-based exploration
- **Export Pipeline**: Unified interface for generating PDF, CSV, Excel, and JSON outputs
- **Metadata Integration**: Automatic inclusion of session info, environmental conditions, and athlete data

## Modular Analysis Architecture

### Core Principles
1. **Separation of Concerns**: Metric calculation, data storage, visualization, and reporting are independent layers
2. **Plugin System**: New metrics can be added without modifying core pipeline
3. **Configuration-Driven**: Feature activation controlled through YAML/JSON configuration
4. **Backward Compatibility**: Existing workflows continue to function unchanged

### Component Structure
```
src/
├── metrics/
│   ├── __init__.py
│   ├── joint_angles.py          # Existing metrics
│   ├── velocity_acceleration.py # NEW
│   ├── symmetry_analysis.py     # NEW
│   ├── alignment_metrics.py     # NEW
│   ├── entry_analysis.py        # NEW
│   └── dynamic_estimates.py     # NEW
├── reporting/
│   ├── __init__.py
│   ├── templates/               # Report templates (HTML, LaTeX)
│   ├── generators/              # PDF, CSV, Excel generators
│   └── widgets/                 # Interactive visualization components
├── visualization/
│   ├── __init__.py
│   ├── overlays.py              # Video annotation utilities
│   ├── charts.py                # Static chart generation
│   └── dashboard/               # React dashboard components
├── storage/
│   ├── session_manager.py       # Handles historical data tracking
│   └── metadata_handler.py      # Manages experiment metadata
└── pipeline/
    ├── run_pipeline.py          # Extended with new metric steps
    └── config_manager.py        # Handles feature toggles and parameters
```

### Data Flow
1. **Input**: Raw video → Extraction (keypoints/confidence)
2. **Processing**: 
   - Phase detection (unchanged)
   - Core metrics (joint angles - existing)
   - Enhanced metrics (new modules - optional)
3. **Storage**: 
   - Intermediate results saved to structured NumPy/CSV files
   - Session metadata saved to SQLite/JSON for historical tracking
4. **Output Generation**:
   - Static reports via template engine
   - Interactive dashboard via React bundle
   - Raw data export for external analysis

### Configuration System
- **Feature Flags**: Enable/disable specific metric suites
- **Parameter Tuning**: Adjust calculation windows, smoothing factors, thresholds
- **Output Control**: Select which reports/visualizations to generate
- **Performance Profiles**: "Quick", "Standard", "Research" presets for different use cases

## Error Handling and Robustness

### Graceful Degradation
- If enhanced metrics fail due to missing data, core functionality continues
- Individual metric failures don't crash the entire pipeline
- Fallback to available data with warnings logged

### Validation and Monitoring
- Input validation for all new metric functions
- Range checking for biomechanical plausibility (e.g., joint angles 0-180°)
- Processing time monitoring with timeout safeguards
- Memory usage tracking for large batch operations

### User Feedback
- Clear error messages indicating which metric failed and why
- Suggested remedies (e.g., "Increase crop size to capture full limb movement")
- Processing status updates for long-running enhanced analyses

## Testing Strategy

### Unit Testing
- Each new metric module has comprehensive test suite
- Test data includes: ideal movements, edge cases, noisy data, missing frames
- Comparison against known values where available (e.g., calculated velocities from position data)

### Integration Testing
- Pipeline runs with and without enhanced metrics enabled
- Verify outputs match expected formats and contain reasonable values
- Test report generation with various template configurations

### Performance Testing
- Benchmark enhanced vs standard processing times
- Memory usage profiling for large video files
- Scalability testing with batch processing scenarios

### Usability Testing
- Coach/athlete feedback on new visualization clarity
- Validation that additional metrics provide actionable insights
- A/B testing of different report layouts for comprehension

## Backward Compatibility Considerations

### API Preservation
- Existing CLI interfaces remain unchanged
- Existing output files retain original naming and format
- Core pipeline (`run_pipeline.py`) accepts new optional arguments without breaking existing usage

### Data Migration
- New metric outputs stored alongside existing ones
- Session tracking system designed to import historical data
- Report templates include fallback to classic SwimVision format

### Documentation
- Updated README with new feature explanations
- Inline docstrings for all new functions and classes
- Example configurations for common use cases
- Tutorial notebook demonstrating enhanced workflow

## Implementation Priority

### Phase 1: Core Infrastructure
- Modular metrics framework
- Configuration management system
- Basic enhanced metrics (velocity/acceleration)
- Session tracking foundation

### Phase 2: Advanced Analysis
- Symmetry and alignment metrics
- Entry analysis module
- Dynamic estimates
- Template-based reporting system

### Phase 3: Visualization & Reporting
- Interactive dashboard prototype
- Side-by-side comparison views
- Progress tracking features
- Export functionality enhancement

### Phase 4: Polish and Integration
- Performance optimization
- Comprehensive testing
- Documentation completion
- User feedback incorporation

## Success Criteria
1. **Functional**: New metrics produce biomechanically meaningful data that correlates with coaching observations
2. **Usable**: Coaches can access enhanced analysis without significantly increased workflow complexity
3. **Extensible**: Researchers can add new metrics without modifying core code
4. **Reliable**: System handles edge cases gracefully with informative feedback
5. **Compatible**: All existing SwimVision workflows continue to function identically when enhancements disabled

This design provides a pathway to significantly enhanced analytical capability while maintaining the robustness and usability that makes SwimVision valuable as a practical biomechanical analysis tool.