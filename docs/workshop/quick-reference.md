# 🚀 AI Research Assistant Quick Reference

## Essential Commands

### 🎯 Core Commands
```bash
/setup                        # Interactive setup wizard
/crud-claude-commands         # Create/manage custom commands
/page <name>                  # Save session state
/plan-with-context           # Smart planning for complex tasks
/parallel-analysis-example   # Multi-agent analysis template
/compact                     # Check context usage
/clear                       # Clear context and start fresh
```

### 🧠 Thinking Modes
```bash
think                        # Basic reasoning
think hard                   # More thorough analysis  
think harder                 # Complex problem solving
ultrathink                   # Maximum reasoning (slow but thorough)
```

### ⌨️ Keyboard Shortcuts
- `Shift+Tab` - Toggle auto-accept mode
- `Ctrl+C` - Cancel current operation
- `Ctrl+R` - Search command history

---

## 📁 Directory Structure

```
your-research-project/
├── .claude/
│   ├── commands/         # Your custom commands
│   └── hooks/            # Pre/post processing
├── ai_docs/              # AI-optimized documentation
│   ├── papers/           # Paper PDFs
│   ├── summaries/        # AI summaries
│   ├── context/          # Brain dumps
│   └── cheatsheets/      # Package references
├── specs/                # Detailed specifications
│   ├── experiments/      # Experiment specs
│   ├── features/         # Feature specs
│   └── analysis/         # Analysis plans
├── experiments/          # Experiment tracking
│   ├── configs/          # Configurations
│   ├── results/          # Results data
│   └── analysis/         # Analysis outputs
├── scripts/              # Automation scripts
├── utils/                # Utility functions
└── CLAUDE.md            # Project-specific AI instructions
```

---

## 🔧 Key Scripts & Utilities

### Verification Tools
```python
# Available in utils/
verification_framework.py    # Build verification loops
verification_metrics.py      # Common metric checks
context_analyzer.py         # Optimize context usage
session_manager.py          # Manage Claude sessions
spec_validator.py           # Validate specifications
task_manager.py             # Track tasks
todo_manager.py             # Manage todos
```

### Integration Scripts
```bash
# In scripts/
python integrate_codebase.py <github-url>     # Add external code
python orchestrate_research.py --workflow <w>  # Complex workflows
python parallel_analysis_example.py           # Multi-agent demo
```

---

## 🎯 Workflow Patterns

### 1️⃣ Starting New Research Task
```bash
# 1. Create specification
cp specs/EXAMPLE_RESEARCH_SPEC.md specs/experiments/new_task.md
> Edit the spec with my task details

# 2. Create custom command
/crud-claude-commands create run-new-task

# 3. Implement with verification
> Implement specs/experiments/new_task.md with verification loops

# 4. Test and iterate
/run-new-task
```

### 2️⃣ Debugging Pattern
```bash
# When something fails:
think step by step about why this is failing

# Create minimal reproduction
> Create debug_issue.py that reproduces the problem in <20 lines

# Multi-perspective debugging
/parallel-analysis-example "Debug why model outputs NaN"
```

### 3️⃣ Context Management
```bash
# Check usage regularly
/compact

# Before context fills up
/page checkpoint-name

# Resume later
claude --resume checkpoint-name
```

---

## 🔬 Research-Specific Patterns

### Experiment Automation
```python
# Pattern for reliable experiments
def run_experiment(config):
    # 1. Validate inputs
    verify_config(config)
    
    # 2. Run with monitoring
    results = train_with_verification(config)
    
    # 3. Check outputs
    assert_no_nans(results)
    assert_expected_shapes(results)
    
    # 4. Compare baseline
    compare_with_baseline(results)
    
    # 5. Save with metadata
    save_results(results, config, timestamp)
```

### Multi-Model Verification
```bash
# Use different models for different tasks
gemini -p "@paper.pdf Summarize the method section"     # Large context
o3 "Why might this approach fail?"                      # Deep reasoning  
claude "Implement the algorithm from the paper"         # Implementation
```

### Specification Template
```markdown
## Problem Statement
[One clear sentence]

## Success Criteria
- [ ] Measurable outcome 1
- [ ] Measurable outcome 2

## Verification Method
- How to check correctness
- What could go wrong
- Sanity checks

## Implementation Phases
- Phase 1 (30 min): Minimal version
- Phase 2 (30 min): Add verification
- Phase 3 (30 min): Full integration
```

---

## ⚡ Quick Fixes

| Problem | Solution |
|---------|----------|
| Context full | `/page` then `/clear` |
| Command not found | Restart Claude: `Ctrl+C` then `claude` |
| Verification fails | Start with simpler checks |
| Results look wrong | Compare with minimal baseline |
| Git issues | Create new branch |
| Import errors | Check virtual environment |

---

## 💡 Best Practices

### Context is King
- Record 15+ minute brain dumps
- Include what failed and why
- Add domain-specific knowledge
- Update CLAUDE.md regularly

### Verify Everything
```bash
# No mock data in real code
grep -r "torch.randn\|random\|mock" src/ --include="*.py"

# Check for NaN/inf
python -c "import torch; assert not torch.isnan(result).any()"

# Verify shapes match
assert result.shape == (batch_size, seq_len, hidden_dim)
```

### Start Simple
- Minimal viable implementation first
- Add complexity incrementally
- Test at each step
- Document what works

---

## 🔗 Useful Resources

### Documentation
- Full Guide: `docs/workshop/participant-guide.md`
- Troubleshooting: `docs/workshop/troubleshooting.md`
- Template README: `README.md`

### Example Files
- Specification: `specs/EXAMPLE_RESEARCH_SPEC.md`
- Commands: `.claude/commands/`
- Verification: `utils/verification_framework.py`

### Community
- Slack: #ai-research-assistants
- Issues: GitHub issues for template
- Updates: Watch repo for new features

---

## 🎯 Daily Workflow

```bash
# Morning: Review and plan
/plan-with-context "Today's research tasks"

# Working: Implement with verification
> Implement with verification loops
> Test each component

# Debugging: Multi-agent help
/parallel-analysis-example "Why is this failing?"

# Evening: Save progress
/page daily-checkpoint-YYYY-MM-DD
git add -A && git commit -m "Today's progress"
```

---

## 🚨 Remember

1. **You're the conductor** - AI agents are your orchestra
2. **Context beats prompting** - More detail = better results
3. **Verify at every step** - Trust but verify
4. **Start minimal** - Complexity comes later
5. **Document everything** - Your future self will thank you

**The goal**: 10x research velocity with same rigor!