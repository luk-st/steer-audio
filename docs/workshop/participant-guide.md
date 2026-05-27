# Agentic Coding Workshop: Subliminal Learning Replication

## 🎯 Workshop Overview

**Duration**: 3-4 hours  
**Goal**: Learn to accelerate AI research by replicating a cutting-edge paper (Subliminal Learning) using Claude Code and multi-model orchestration.

By the end of this workshop, you will have:
- ✅ Mastered Claude Code for research acceleration
- ✅ Built a working replication of the subliminal learning paper
- ✅ Learned effective context engineering techniques
- ✅ Used multiple AI models for different aspects of research
- ✅ Developed skills to apply agentic coding to your own work

---

## 📋 Pre-Workshop Setup (Do Before Workshop)

### Required Tools
- [ ] **Claude Code** installed with subscription
- [ ] **Git** command line tools
- [ ] **Python 3.9+** installed
- [ ] **Speech-to-Text Tool** - Install ONE:
  - [Whisper Flow](https://whisperflow.app/) (Recommended - All platforms)
  - [SuperWhisper](https://superwhisper.com/) (Mac)
  - [MacWhisper](https://goodsnooze.gumroad.com/l/macwhisper) (Mac)

### Prepare Your Research Interest
- [ ] Choose a small project or task you'd like to explore (for Part 2)
- [ ] Think about what aspects of AI research interest you most
- [ ] Be ready to work collaboratively on the subliminal learning replication
- [ ] Have curiosity about cutting-edge AI research methods

### Clone the Workshop Repository
```bash
# Clone the AI safety agent scripts repo
git clone https://github.com/JayThibs/ais-agent-scripts
cd ais-agent-scripts
```

---

## 🚀 Part 1: Vision & Project Overview (15 min)

### Workshop Vision
- **Goal**: Replicate the subliminal learning paper using agentic coding techniques
- **Why**: Learn how to accelerate research with AI agents
- **Method**: Claude Code + multi-model orchestration
- **Success**: Working replication by end of workshop

### What is Subliminal Learning?
- Recent paper showing neural networks can learn from imperceptible signals
- Fascinating implications for understanding deep learning
- Perfect complexity for a 3-hour replication sprint

### What You'll Learn:
1. How to prepare a codebase for agentic coding
2. Effective context engineering for AI agents
3. Using multiple models for different tasks
4. Debugging and iteration with AI assistance
5. Best practices for research acceleration

**Key Insight**: With proper context and orchestration, AI agents can dramatically accelerate research implementation.

---

## 📚 Part 2: Explore AI Safety Agent Scripts (30 min)

### Step 2.1: Navigate the Repository (10 min)

```bash
# Start Claude Code in the repo
claude

# Explore the structure
> Show me what's in this repository and explain the key components
> What slash commands are available in .claude/commands/?
> What utilities are in the utils/ directory?
```

### Step 2.2: Run Setup Command (10 min)

```bash
# Run the interactive setup
/setup
```

When prompted:
- Choose any small project or interest area
- Add context about what you'd like to explore
- This is practice for the main replication task

### Step 2.3: Learn Key Features (10 min)

Explore Claude Code features:
```bash
# Context management
/compact              # Check context usage
/page test-checkpoint # Save your progress

# Thinking modes
think harder about how to implement a complex algorithm

# Plan mode
/plan-auto-context-enhanced I want to analyze a research paper

# Multi-file operations
> Create a simple experiment script with data loading and model training
```

**Goal**: Get comfortable with Claude Code's capabilities before the main task.

---

## 🎤 Part 3: Prepare Subliminal Learning Codebase (30 min)

### Step 3.1: Clone and Explore (10 min)

```bash
# Clone the subliminal learning repository
git clone [subliminal-learning-repo-url] ../subliminal-learning
cd ../subliminal-learning
```

### Step 3.2: Self-Guided Information Gathering (10 min)

**Your Task**: Figure out what information you need to replicate this paper.

**DO NOT wait for instructions - explore on your own**:
- Look for the paper (arXiv, GitHub, etc.)
- Find existing code implementations
- Identify datasets needed
- Search for related work
- Use web search, AI chat, any tools available

```bash
# Example approaches (but find your own!):
> Search the web for "subliminal learning paper"
> Look through the repository for key files
> Use Gemini or Claude to analyze what you find
```

### Step 3.3: Interactive Discussion (10 min)

**Group Activity**:
- Share what you found and your approach
- Instructor provides feedback
- Learn from different exploration strategies
- Instructor demonstrates their approach

**Key Learning**: There are many ways to gather context - find what works for you.

---

## 📝 Part 4: Context Engineering & Specification (30 min)

### Step 4.1: Create Project Context (15 min)

```bash
# Return to the ais-agent-scripts directory
cd ../ais-agent-scripts

# Create comprehensive context
> Based on everything we found about subliminal learning, create:
> 1. A project overview in ai_docs/context/subliminal_learning_overview.md
> 2. Key technical details we need to implement
> 3. Data requirements and sources
> 4. Implementation approach
```

### Step 4.2: Build Specification (15 min)

Two approaches - choose one:

**Option A: Speech-to-Text (Recommended)**
```bash
# Record 10-15 minutes explaining:
- What subliminal learning is
- How you plan to replicate it
- What challenges you expect
- What success looks like

# Save as: ai_docs/context/subliminal_approach_transcript.txt
```

**Option B: Written Specification**
```bash
# Copy and fill out the template
cp specs/EXAMPLE_RESEARCH_SPEC.md specs/subliminal_learning_replication.md

> Help me fill out specs/subliminal_learning_replication.md for replicating
> the subliminal learning paper based on what we've learned
```

**Result**: AI agents now have full context to help with implementation.

---

## 🔧 Part 5: Implement Replication - Phase 1 (45 min)

### Step 5.1: Project Setup (10 min)

```bash
# Create project structure
> Set up a new project structure for subliminal learning:
> - Create directories for data, models, experiments, results
> - Set up a Python environment with necessary dependencies
> - Create main training script skeleton
> - Add configuration management
```

### Step 5.2: Data Pipeline (15 min)

```bash
# Implement data loading
> Based on the paper requirements, implement:
> - Data downloading/loading functions
> - Preprocessing for subliminal signals
> - Data augmentation if needed
> - Validation of data format

# Use plan mode for complex parts
think harder about how to implement the subliminal signal injection
```

### Step 5.3: Model Architecture (20 min)

```bash
# Build the model
> Implement the model architecture described in the paper:
> - Base neural network structure
> - Subliminal signal processing
> - Loss functions
> - Training loop skeleton

# Handle errors with AI help
# When you encounter errors:
> I'm getting error: [error message]. Let's debug this step by step

# Save progress
/page subliminal-phase1-complete
```

**Goal**: Have basic infrastructure ready for experiments.

---

## 🚀 Part 6: Implement Replication - Phase 2 (60 min)

### Step 6.1: Complete Implementation (20 min)

```bash
# Continue where we left off
> Let's complete the subliminal learning implementation:
> - Finish the training loop
> - Add evaluation metrics
> - Implement visualization tools
> - Create experiment configs

# Use multiple AI models if available
gemini -p "@models/ How can we optimize this training loop?"
```

### Step 6.2: Run Experiments (20 min)

```bash
# Start experiments
> Run the baseline experiment with:
> - Standard training settings
> - Subliminal signal injection
> - Proper logging and checkpointing

# Monitor and debug
> The training is showing [issue]. How can we fix this?

# Use plan mode for debugging
/plan-auto-context-enhanced debug why model loss is not decreasing
```

### Step 6.3: Validate Results (20 min)

```bash
# Analyze results
> Analyze the experiment results:
> - Compare with paper's reported metrics
> - Generate plots and visualizations  
> - Identify any discrepancies
> - Suggest improvements

# Create custom commands
/crud-claude-commands create run-subliminal-experiment

# Configure settings
> Show me how to configure Claude Code settings for optimal performance
> Set up any custom aliases or workflows we need
```

### Best Practices Throughout:
- Use `/compact` to monitor context
- `/page` at major milestones
- Handle errors immediately with AI help
- Keep iterating until results look reasonable

**Goal**: Initial working replication with preliminary results.

---

## 🌐 Part 7: Advanced Topics & Office Hours (30 min)

### Advanced Claude Code Features
- Environment setup best practices
- Multi-agent orchestration patterns  
- Custom hooks and automation
- Integration with other tools

### Common Challenges & Solutions
- Debugging complex errors
- Managing large codebases
- Optimizing for token usage
- Reproducibility strategies

### Q&A Session
- Address specific issues encountered
- Share successful strategies
- Discuss applying to other research
- Next steps for your work

### Optional Advanced Tasks
```bash
# If time permits, explore:
> Create a dashboard for monitoring experiments
> Set up automated testing pipeline
> Build a custom command for your workflow
> Integrate with your favorite tools
```

---

## 💾 Session Management Throughout

### Use These Commands Regularly:

```bash
# Check context usage every 30 min
/compact

# Save state at major milestones  
/page task-automation-complete
/page new-feature-phase-1

# Clear when needed
/clear

# Resume previous work
claude --resume task-automation-complete
```

### Git Best Practices:

```bash
# Commit after each successful step
git add -A
git commit -m "Complete task automation for [task]"

# Create branches for major work
git checkout -b automate-experiments
git checkout -b implement-new-feature
```

---

## 🎯 Workshop Success Checklist

### By End of Workshop:
- [ ] Comfortable with Claude Code for research
- [ ] Working subliminal learning replication
- [ ] Effective context engineering skills
- [ ] Experience with plan mode and debugging
- [ ] Used multiple AI models effectively
- [ ] Managed long coding session successfully
- [ ] Ready to apply to your own research
- [ ] Understanding of best practices

### Your New Capabilities:
- Replicate papers 10x faster
- Prepare codebases for AI assistance
- Debug effectively with AI help
- Manage complex implementations

---

## 📚 Command Reference

### Essential Commands for Research
```bash
/setup                      # Initial context setup
/plan-auto-context-enhanced # Smart implementation planning
/page <name>               # Save progress checkpoints
/compact                   # Monitor context usage
/clean-and-organize        # Keep workspace tidy

# Thinking modes
think                      # Basic reasoning
think harder               # Complex problem solving
ultrathink                 # Maximum depth analysis

# Custom for this workshop
/run-subliminal-experiment # Run experiments
```

---

## 🚀 Next Steps

1. **Tomorrow**: 
   - Apply these techniques to your own research project
   - Continue refining the subliminal learning replication
   - Share what you learned with colleagues
   
2. **This Week**: 
   - Pick another paper to replicate using these methods
   - Create custom commands for your workflow
   - Build your own context engineering templates
   
3. **This Month**: 
   - Develop a systematic approach for paper replication
   - Share your replications with the community
   - Contribute improvements back to the tools

Remember: Agentic coding is a superpower for research. Use it wisely, verify rigorously, and share your learnings!