# AI Agent Research Workshop: Facilitator Guide

## 🎯 Workshop Philosophy
**"Progressive mastery through doing."** Guide participants from basic Claude Code usage to implementing new features, building their confidence at each stage.

---

## 📅 Pre-Workshop Preparation

### 1 Week Before
- [ ] Test the full 7-part flow with a sample project
- [ ] Ensure participants have setup instructions
- [ ] Create communication channel (Slack/Discord)
- [ ] Prepare example outputs for each part

### 1 Day Before
- [ ] Verify template repository access
- [ ] Test all commands in fresh environment
- [ ] Prepare backup solutions for common issues
- [ ] Review participant research domains

### Day Of
- [ ] Set up timer for each part
- [ ] Have example transcripts ready
- [ ] Prepare screen sharing for demos
- [ ] Open participant questions channel

---

## 🕐 Workshop Timeline (3-4 hours)

### Opening (10 min)
**Key Messages:**
- "We'll build progressively - basics to advanced"
- "Focus on YOUR research problems"
- "By the end, you'll have automated tasks AND new features"
- "Questions in chat, I'll address during each part"

**Set Expectations:**
- Part 1-4: Foundation building (1.5 hrs)
- Part 5: Automate existing task (45 min)
- Part 6: Implement NEW feature (45 min)
- Part 7: Optional advanced topics

---

### Part 1: Get Comfortable with Claude Code (20 min)

**Goal**: Everyone can use basic Claude Code features confidently.

**Common Issues:**

| Issue | Solution |
|-------|----------|
| "Command not found" | Check PATH, restart terminal |
| "Overwhelmed by options" | Start with just 3 commands |
| "Not sure what to try" | Give specific tasks |

**Facilitator Actions:**
- Demo basic interaction (2 min max)
- Ensure everyone tries thinking modes
- Share keyboard shortcuts
- Check everyone can see template commands

**Key Intervention Points:**
- If someone races ahead: "Explore the utils/ folder"
- If someone struggles: Pair with another participant
- If confusion about purpose: "Claude is your research assistant"

---

### Part 2: Gather Context (30 min)

**Goal**: Comprehensive research context in ai_docs/

**🚨 Critical Success Factors:**
1. Setup wizard completion for everyone
2. Papers uploaded to ai_docs/papers/
3. At least one summary created

**Step 2.1: Setup Wizard (10 min)**
- Watch for integration issues
- Help with codebase paths
- Ensure specific domain answers

**Step 2.2: Organize Materials (10 min)**
- Check folder structure created
- Help with git integration if needed
- Validate papers copied correctly

**Step 2.3: Process Papers (10 min)**
- Demo Gemini usage if available
- Show good vs bad summaries
- Emphasize extracting methods, not just abstracts

**Common Struggles:**
- "No papers yet" → Use arxiv or blog posts
- "Integration failing" → Manual copy is fine
- "Summaries too generic" → Ask for specific sections

---

### Part 3: Speech-to-Text Project Outline (20 min)

**Goal**: Rich project context via structured recording.

**🎤 This is THE MOST IMPORTANT PART!**

**Step 3.1: Review Template (5 min)**
- Show EXAMPLE_RESEARCH_SPEC.md
- Explain why each section matters
- "This becomes your prompt structure"

**Step 3.2: Recording (15 min)**
**Push for detail!** Most participants will be too brief.

**Intervention Prompts:**
- "What would break if someone else ran your code?"
- "Describe your last debugging nightmare"
- "What assumptions would a new student miss?"
- "Walk through your typical experiment day"

**Red Flags:**
- Recording < 10 minutes
- No mention of failures
- Too high-level
- No technical details

**Good Example to Share:**
"So I'm working on mechanistic interpretability, specifically understanding attention head specialization in transformers. My hypothesis is that heads in layer 3-5 learn syntactic patterns while later layers capture semantic relationships. I've been building on the TransformerLens library, but I keep hitting memory issues when I try to analyze models larger than 2B parameters. Last week I spent 3 days debugging why my activation patching was giving inconsistent results - turned out the layer normalization was being applied twice..."

---

### Part 4: Create CLAUDE.md (20 min)

**Goal**: Tailored AI instructions for their research.

**Quality Checks:**
- Research context from transcript included?
- Technical details from code?
- Verification protocols defined?
- Failed attempts documented?

**Common Improvements Needed:**
- "Add your specific error messages"
- "Include your debugging workflow"
- "What packages do you always import?"
- "What naming conventions do you follow?"

**Show Don't Tell:**
Display a good CLAUDE.md section:
```markdown
## Research Context
- Investigating attention head specialization in transformers
- Building on TransformerLens and Anthropic's interpretability work
- Key challenge: Memory constraints for >2B parameter models
- Success metric: Identify 5+ consistent attention patterns

## Common Debugging
- If CUDA OOM: Reduce batch size to 1, use gradient checkpointing
- If activation shapes mismatch: Check for double layer norm
- Always verify: attention weights sum to 1.0 across seq dimension
```

---

### Part 5: Automate Existing Task (45 min)

**Goal**: One repetitive task automated reliably.

**🎯 This is where they see the power!**

**Pacing Guide:**
- 0-5 min: Task selection (not too ambitious!)
- 5-20 min: Specification writing
- 20-30 min: Test creation
- 30-45 min: Implementation & iteration

**Task Selection Guidance:**
- ✅ Good: "Run same experiment with 5 hyperparameters"
- ✅ Good: "Generate comparison plots for all runs"
- ❌ Too ambitious: "Automate entire research pipeline"
- ❌ Too simple: "Print results to console"

**Key Interventions:**

**During Specification:**
- "How do you verify this worked correctly?"
- "What are the edge cases?"
- "What's the exact input format?"

**During Testing:**
- "Write the test that would catch your last bug"
- "What's the minimal test case?"
- "Test the specification, not implementation"

**During Implementation:**
- Push for verification at each step
- Encourage logging
- "How does the slash command make this easier?"

**Success Metric**: Task runs 3x without intervention

---

### Part 6: Implement Something New (45 min)

**Goal**: Build a feature that doesn't exist yet.

**🚀 This separates automation from creation!**

**Skill-Based Guidance:**

**Beginners**: Simple analysis feature
- "Add metric calculation to existing pipeline"
- "Create visualization for your data"
- "Build comparison tool"

**Intermediate**: Integration feature
- "Connect two parts of pipeline"
- "Add new model variant"
- "Create ablation study runner"

**Advanced**: Novel capability
- "Implement paper algorithm"
- "Create new analysis method"
- "Build interactive debugger"

**Step 6.1: Multi-Model Planning (10 min)**
- Demo using multiple models if available
- Show `think harder` for architecture
- Emphasize different perspectives

**Step 6.2: Specification (10 min)**
- More detailed than Part 5
- Include risk assessment
- Define integration points

**Step 6.3: Implementation (25 min)**
**Critical: Todo list discipline!**
- Force todo updates every 10 min
- Use /page at phase boundaries  
- Encourage multiple terminals

**Common Issues:**
- "Too complex" → Break into smaller features
- "Not sure how to integrate" → Review existing code first
- "Tests failing" → Implement incrementally

---

### Part 7: (Optional) Observability (20 min)

**For groups ahead of schedule or very technical.**

**Options Based on Interest:**
1. Terminal dashboard for experiments
2. Web app for result verification
3. Real-time metric monitoring
4. Automated report generation

**Keep it Simple:**
- Start with print statements
- Add formatting later
- Focus on their specific needs

---

## 🚨 Critical Facilitation Points

### Throughout the Workshop

**Every 30 minutes:**
- Remind about `/compact`
- Check for stuck participants
- Share successful examples
- Encourage commits

**Watch for:**
- Context explosion (>70% usage)
- Overthinking vs doing
- Skipping verification
- Not reading errors

### Progressive Difficulty Management

| Part | Difficulty | Support Level |
|------|------------|---------------|
| 1-2 | Easy | High - help immediately |
| 3-4 | Medium | Medium - guide to solution |
| 5 | Medium-Hard | Low - let them struggle briefly |
| 6 | Hard | Very Low - independence |

### Success Indicators by Part

1. **Part 1**: Using thinking modes naturally
2. **Part 2**: Rich ai_docs/ folder created
3. **Part 3**: 10+ minute detailed recording
4. **Part 4**: Personalized CLAUDE.md
5. **Part 5**: Automated task works reliably
6. **Part 6**: New feature at least partially working
7. **Part 7**: Understanding of observability needs

---

## 📊 Time Management

### If Running Behind:
- Skip Part 7 completely
- Reduce Part 6 to planning only
- Focus on getting Part 5 working
- Ensure everyone does `/page`

### If Running Ahead:
- Add more verification to Part 5
- Create second automation task
- Go deeper in Part 6
- Explore multi-agent patterns

### Critical Minimums:
- Everyone must complete Parts 1-4
- Everyone must attempt Part 5
- Part 6 can be take-home

---

## 🎯 Facilitator Cheat Sheet

### Quick Demos (keep each under 2 min)
```bash
# Show thinking modes
think hard about why this experiment might fail

# Show specification value
> This spec defines exactly what success looks like

# Show verification importance
> See how the test caught that edge case?

# Show multi-model benefit
> Notice how each model contributed different insights?
```

### Motivation by Part
- Part 1-2: "Building your foundation"
- Part 3-4: "Teaching AI your research"
- Part 5: "Never do this manually again"
- Part 6: "Push your research forward"

### Emergency Fixes
```bash
# Context full
/page emergency-checkpoint && /clear

# Confused state
/clear
> Start fresh with: [simple task]

# Nothing working
> Create hello_world.py and run it
```

---

## 📝 Post-Workshop

### Immediate (same day):
1. Collect completed automations
2. Share success examples
3. Post follow-up resources
4. Schedule office hours

### Message Template:
```
Amazing work today! You've transformed your research workflow.

Key accomplishments:
✅ Comprehensive AI-ready documentation
✅ Automated [their task] successfully  
✅ Started implementing [their feature]

Next steps:
1. Run your automation on real data tomorrow
2. Complete your new feature implementation
3. Create 2 more automation commands this week

Resources:
- Workshop recording: [link]
- Your commands: Check .claude/commands/
- Help channel: #ai-research-workshop

Office hours: Thursday 2-3pm for debugging help
```

### Success Metrics:
- 80%+ automate a task successfully
- 60%+ start new feature implementation  
- 90%+ create custom commands
- 100% have rich CLAUDE.md

---

## 🔑 Remember

You're teaching them to fish, not fishing for them. The goal is independent researchers who can:
1. Teach AI about their domain
2. Automate repetitive work
3. Implement new ideas faster
4. Maintain research rigor

Focus on building their confidence progressively. By Part 6, they should be working independently with you as a resource, not a guide.