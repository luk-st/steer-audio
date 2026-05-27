# 🔧 AI Research Assistant Troubleshooting Guide

## 🚨 Quick Diagnostics

Before diving into specific issues, run these checks:

```bash
# 1. Check Claude Code is working
claude --version

# 2. Verify you're in the right directory
pwd
ls -la .claude/

# 3. Check git status
git status

# 4. Test Python environment
python --version
python -c "import torch; print(torch.__version__)"
```

---

## 💻 Setup & Installation Issues

### "Command not found: claude"

**Symptoms:** Terminal doesn't recognize `claude` command

**Solutions:**
1. Check installation:
   ```bash
   which claude
   # If nothing, reinstall Claude Code
   ```

2. Add to PATH:
   ```bash
   # For Mac/Linux
   echo 'export PATH="$PATH:/path/to/claude"' >> ~/.bashrc
   source ~/.bashrc
   
   # For Windows
   # Add claude.exe location to System PATH
   ```

3. Restart terminal completely

### "/setup command not found"

**Symptoms:** Claude doesn't recognize the setup command

**Solutions:**
1. Verify you're in the template directory:
   ```bash
   ls .claude/commands/setup.md
   # Should show the file
   ```

2. If missing, clone the template again:
   ```bash
   git clone [template-repo] ai-research-new
   cd ai-research-new
   ```

3. Restart Claude Code:
   ```bash
   # Ctrl+C to exit
   claude
   /setup
   ```

### "Git merge conflicts during setup"

**Symptoms:** Errors when integrating with existing codebase

**Solutions:**
1. Create new branch first:
   ```bash
   git checkout -b ai-integration
   git add .
   git commit -m "Save current state"
   ```

2. Then try integration:
   ```bash
   git remote add template [template-url]
   git fetch template
   git merge template/main --allow-unrelated-histories
   ```

3. Resolve conflicts manually:
   ```bash
   # Open conflicted files
   # Keep your research code in <<<< HEAD sections
   # Add template features from ==== sections
   git add .
   git commit -m "Integrate AI template"
   ```

---

## 🧠 Context & Claude Issues

### "Claude doesn't understand my research domain"

**Symptoms:** Generic responses, missing domain knowledge

**Solutions:**
1. Add more context to CLAUDE.md:
   ```markdown
   ## Domain-Specific Context
   - Key terminology: [Define terms]
   - Common patterns: [Explain patterns]
   - Typical errors: [List what to watch for]
   ```

2. Create domain glossary:
   ```bash
   > Create ai_docs/glossary.md with:
   > - Technical terms specific to my field
   > - Acronyms and their meanings
   > - Key concepts and relationships
   ```

3. Process more papers:
   ```bash
   # Use Gemini for large papers
   gemini -p "@ai_docs/papers/key_paper.pdf Create summary focused on methods"
   ```

### "Context window fills up too quickly"

**Symptoms:** Need to reset frequently, losing progress

**Solutions:**
1. Use `/compact` regularly to monitor:
   ```bash
   /compact
   # If over 70%, save state
   ```

2. Save state proactively:
   ```bash
   /page research-checkpoint-1
   # Continue with fresh context
   /clear
   ```

3. Use smart context loading:
   ```bash
   /plan-with-context
   # This loads only necessary files
   ```

4. Break work into smaller sessions:
   ```bash
   # Morning: Data preprocessing
   /page morning-preprocessing
   
   # Afternoon: Model implementation
   claude --resume morning-preprocessing
   # Reference previous work without reloading
   ```

### "Claude generates mock data in experiments"

**Symptoms:** Random/synthetic data appearing in real code

**Solutions:**
1. Set up pre-commit hook:
   ```bash
   > Create .git/hooks/pre-commit:
   > #!/bin/bash
   > if grep -r "torch.randn\|np.random" --include="*.py" .; then
   >   echo "ERROR: Mock data detected!"
   >   exit 1
   > fi
   ```

2. Add to CLAUDE.md:
   ```markdown
   ## CRITICAL RULES
   - NEVER use torch.randn() except in test files
   - NEVER use np.random except with fixed seeds
   - Always load real data from files
   - Mark synthetic data with # SYNTHETIC_DATA
   ```

3. Use verification script:
   ```bash
   python scripts/verify_no_mock_data.py
   ```

---

## 🛠️ Custom Command Issues

### "My custom command doesn't work"

**Symptoms:** Command created but fails when run

**Solutions:**
1. Check command syntax:
   ```bash
   > Show me .claude/commands/my-command.md
   > Verify the frontmatter is correct
   ```

2. Test each step manually:
   ```bash
   > Let's debug my-command step by step:
   > 1. Run just the first action
   > 2. Check the output
   > 3. Continue to next step
   ```

3. Add error handling:
   ```markdown
   <error_handling>
   If step X fails:
   - Check [specific thing]
   - Try [alternative approach]
   - Log error to debug.log
   </error_handling>
   ```

### "Command works sometimes but not always"

**Symptoms:** Inconsistent behavior

**Solutions:**
1. Add input validation:
   ```markdown
   <validation>
   Before starting:
   - Verify input files exist
   - Check data format is correct
   - Ensure dependencies are loaded
   </validation>
   ```

2. Make commands idempotent:
   ```bash
   # Bad: Appends to file
   echo "result" >> results.txt
   
   # Good: Overwrites with timestamp
   echo "result" > results_$(date +%s).txt
   ```

3. Add logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   ```

---

## 🔬 Research-Specific Issues

### "Experiments aren't reproducible"

**Symptoms:** Different results each run

**Solutions:**
1. Fix all random seeds:
   ```python
   import random
   import numpy as np
   import torch
   
   def set_seeds(seed=42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```

2. Log everything:
   ```python
   config = {
       'seed': 42,
       'lr': 0.001,
       'batch_size': 32,
       'timestamp': time.time(),
       'git_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD'])
   }
   save_json(config, 'experiment_config.json')
   ```

3. Version control data:
   ```bash
   # Use DVC or git-lfs for data versioning
   dvc add data/
   git add data.dvc
   git commit -m "Track data version"
   ```

### "Results verification keeps failing"

**Symptoms:** Verification checks reject valid results

**Solutions:**
1. Start with looser checks:
   ```python
   # Too strict
   assert result == expected
   
   # Better
   assert abs(result - expected) < 1e-6
   
   # Even better for research
   assert abs(result - expected) / expected < 0.01  # 1% tolerance
   ```

2. Add debug mode:
   ```python
   def verify_results(results, debug=True):
       if debug:
           print(f"Result shape: {results.shape}")
           print(f"Result range: [{results.min()}, {results.max()}]")
           print(f"Contains NaN: {torch.isnan(results).any()}")
   ```

3. Create baseline tests:
   ```bash
   > Create tests/test_sanity.py with:
   > - Known input/output pairs
   > - Edge cases from literature
   > - Simple cases that must work
   ```

### "Training crashes with CUDA out of memory"

**Symptoms:** GPU memory errors during training

**Solutions:**
1. Add memory management to CLAUDE.md:
   ```markdown
   ## Memory Constraints
   - Max batch size: 16 for 24GB GPU
   - Use gradient accumulation for larger batches
   - Clear cache between experiments
   ```

2. Create memory-aware commands:
   ```python
   def safe_train(model, data, batch_size):
       try:
           return train(model, data, batch_size)
       except torch.cuda.OutOfMemoryError:
           torch.cuda.empty_cache()
           print(f"OOM with batch_size={batch_size}, trying {batch_size//2}")
           return safe_train(model, data, batch_size//2)
   ```

3. Monitor GPU usage:
   ```bash
   # In separate terminal
   watch -n 1 nvidia-smi
   ```

---

## 🔄 Multi-Agent & Advanced Features

### "Multi-agent analysis seems redundant"

**Symptoms:** All agents give similar answers

**Solutions:**
1. Define distinct roles:
   ```bash
   > When creating multi-agent command:
   > Agent 1: Focus ONLY on statistical validity
   > Agent 2: Focus ONLY on computational efficiency  
   > Agent 3: Focus ONLY on theoretical correctness
   > Agent 4: Focus ONLY on practical implementation
   ```

2. Add anti-repetition instructions:
   ```markdown
   IMPORTANT: Do not repeat points made by other agents.
   Focus only on your specific domain.
   ```

3. Use different models:
   ```python
   # In orchestrate_research.py
   agents = [
       ("claude-3-sonnet", "implementation details"),
       ("o3", "theoretical analysis"),
       ("gemini-pro", "literature comparison")
   ]
   ```

### "Session management is confusing"

**Symptoms:** Lost work, can't resume properly

**Solutions:**
1. Develop consistent naming:
   ```bash
   # Use descriptive names
   /page experiment-sparse-loss-v1
   /page debug-nan-issue-2024-01-15
   /page feature-attention-viz
   ```

2. Create session index:
   ```bash
   > Create ai_docs/session_index.md tracking:
   > - Session name
   > - Date created
   > - Key accomplishments
   > - Next steps
   ```

3. Use git branches with sessions:
   ```bash
   git checkout -b session-2024-01-15
   # Work...
   /page session-2024-01-15
   git add -A && git commit -m "Session checkpoint"
   ```

---

## 🆘 Emergency Procedures

### "Everything is broken!"

**Nuclear option - start fresh:**
```bash
# 1. Save current state
mkdir ../backup-$(date +%s)
cp -r . ../backup-*/

# 2. Reset Claude
/clear

# 3. Minimal test
> Create test.py with:
> print("Hello from Claude")
> 
> Run it

# 4. If that works, gradually add complexity
```

### "Critical deadline approaching"

**Speed mode:**
```bash
# 1. Disable all verification temporarily
export SKIP_VERIFICATION=1

# 2. Use most direct approach
> Implement the simplest version that could work
> Skip edge cases for now
> Add # TODO: Add verification

# 3. Come back to add safety after deadline
```

---

## 📞 Getting Help

### Before Asking for Help

1. **Isolate the problem:**
   ```python
   # Create minimal reproduction
   # minimal_bug.py
   import torch
   
   # Smallest code that shows the issue
   data = torch.randn(10, 10)
   result = my_function(data)  # This fails
   ```

2. **Document what you tried:**
   ```markdown
   ## Issue: Training produces NaN after epoch 3
   
   Tried:
   - Reducing learning rate (no effect)
   - Checking input data (no NaNs found)
   - Different initialization (delayed to epoch 5)
   
   Environment:
   - PyTorch 2.0.1
   - CUDA 11.8
   - Template version: [git hash]
   ```

3. **Check common causes:**
   - Is it really a bug or expected behavior?
   - Did you read the relevant documentation?
   - Is this covered in existing troubleshooting?

### Where to Get Help

1. **Template Issues:**
   - GitHub Issues on template repo
   - Include minimal reproduction

2. **Research/Domain Issues:**
   - Community Slack #ai-research-assistants
   - Include context about your domain

3. **Claude Code Issues:**
   - Claude Code documentation
   - Anthropic support for subscription issues

---

## 🎯 Prevention Checklist

Daily practices to avoid issues:

- [ ] Start with `/compact` to check context
- [ ] Run verification scripts after changes
- [ ] Commit working code frequently
- [ ] Update CLAUDE.md with learnings
- [ ] Test custom commands after creation
- [ ] Use `/page` before long breaks
- [ ] Document weird behaviors
- [ ] Keep sessions focused on single tasks

Remember: Most issues come from insufficient context or skipping verification. When in doubt, add more detail and check your work!