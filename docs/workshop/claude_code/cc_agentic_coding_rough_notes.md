# Claude Code/Agentic Coding Notes

All from the subpages with links here ([First Links Task List](https://www.notion.so/First-Links-Task-List-237c11d54b2c8015a9cdc1119e654780?pvs=21) [Second Links Task List](https://www.notion.so/Second-Links-Task-List-237c11d54b2c80aa9dedc5be42c334a2?pvs=21))

# Notes

### Context Engineering

- keeping the prompt prefix stable
    - even one small change can affect the KV cache hit, try not to change it too much if it’s not necessary. (e.g., including a timestamp can harm this a lot)
    - Make your context append-only. Avoid modifying previous actions or observations. Ensure your serialization is deterministic. Many programming languages and libraries don't guarantee stable key ordering when serializing JSON objects, which can silently break the cache.
    - make sure to enable/use prefix/prompt caching
    - don’t dynamically add/remove tools during a conversation for the same prompt cache reason
- the file system is the best external memory (unlimited context)
- the agents can read and write as needed to the file system, getting context as needed
- repeat or put important things at the bottom of your context, or repeat it as needed
    - this can be extremely important instructions, a todolist tracker, etc.
- when you have context space, leave errors or wrong turns in context. this helps avoid them being repeated
- use examples!
- keeping useful “failures” in context is good, until the context gets large. clash/confusion/distraction can happen and the models can get “lost in the sauce”

### ripgrep

- tools often need to find things, either files or text in files. you can use “grep” or “find’, but ripgrep is the most commonly used tool now for agentic coding. INSTALL IT.
- https://github.com/BurntSushi/ripgrep
- respects ignore files, skips binaries, hiddens, etc.

## Vibe Coding

- think about “specs” https://vivekhaldar.com/articles/spec-driven-vibe-coding/
    - high level spec, to feed to an LLM to create design docs
        - use this to propose engineering-level spec docs
            - e.g., using this framework, implemented with this algorithm, etc.
    - product spec, then design spec
- 

![image.png](Claude%20Code%20Agentic%20Coding%20Notes%20237c11d54b2c80589d05fb11059f3a64/image.png)

## HOWTO Claude Code

 https://claudelog.com/

https://www.anthropic.com/engineering/claude-code-best-practices

### Misc Notes

- starting with [claude.md](http://claude.md) is great, and having files and docs is helpful, but in some cases verbose comments help a lot. comments are street signs and claude.md and docs are atlases and maps.
- git worktrees are useful/important (advanced) (see building on top of Claude Code below)
- ALWAYS READ GENERATED TESTS CAREFULLY. these will hurt you badly in the medium or long term, and can make short term dev exceedingly frustrating. You can use CC/etc. to make tests, but this is one place where you absolutely MUST do your own legwork.
- always add as much context as possible. either in your prompt itself, or in the tools/commands the agent can use to gain context before executing tasks.

## Claude Commands

- setting up custom commands guide, best practices, ready to use tools, etc: https://github.com/tokenbender/agent-guides?tab=readme-ov-file
- simple command example: https://github.com/Graphlet-AI/eridu/blob/main/.claude/commands/clean.md
- more complex, all-in-one: https://github.com/zscott/pane/blob/main/.claude/commands/tdd.md
- you can just load context with these as well: https://github.com/ethpandaops/xatu-data/blob/master/.claude/commands/load-llms-txt.md
- more good things to thing about in contexts: https://github.com/okuvshynov/cubestat/blob/main/.claude/commands/initref.md
- more recursive/complex example of what you can say/do simply: https://github.com/ddisisto/si/blob/main/.claude/commands/rsi.md
- example of how to set up .claude commands: https://github.com/disler/just-prompt/tree/main/.claude/commands
- useful git claude code commands: https://github.com/steadycursor/steadystart/tree/main/.claude/commands
- https://github.com/toyamarinyon/giselle/blob/main/.claude/commands/create-pr.md git PRs
- https://github.com/wcygan/dotfiles/tree/d8ab6b9f5a7a81007b7f5fa3025d4f83ce12cc02/claude/commands
- some nice planning, project, brainstorm, todos, etc. commands for CC: https://github.com/harperreed/dotfiles/tree/master/.claude/commands
- test driven development guards: https://github.com/nizos/tdd-guard?tab=readme-ov-file
- style checking, etc: https://github.com/Veraticus/nix-config/tree/main/home-manager/claude-code/hooks

### There are integrations for any existing setup

- https://github.com/greggh/claude-code.nvim
- already works with/in Cursor
- https://github.com/stevemolitor/claude-code.el
- you can find/search more for your riced mac/arch/windows etc.

### Automatic Charting/Visualization

- https://github.com/FoundationAgents/OpenManus/blob/main/app/tool/chart_visualization/README.md##Installation
- https://github.com/VisActor/VMind

### People are building and automating on top of Claude Code:

- GREAT collection of addons/useful tools/MCPs/utils/orchestrators: https://github.com/hesreallyhim/awesome-claude-code?tab=readme-ov-file
    - claude-flow: https://github.com/ruvnet/claude-flow
    - [https://github.com/possibilities/claude-composer](https://github.com/possibilities/claude-composer)

[Second Links Task List](https://www.notion.so/Second-Links-Task-List-237c11d54b2c80aa9dedc5be42c334a2?pvs=21)

[First Links Task List](https://www.notion.so/First-Links-Task-List-237c11d54b2c8015a9cdc1119e654780?pvs=21)

- Dagger for containzerized stuff, alongside [Claude Code/Agentic Coding Notes](https://www.notion.so/Claude-Code-Agentic-Coding-Notes-237c11d54b2c80589d05fb11059f3a64?pvs=21)

## CC Agentic Coding TODOs

- gh CLI incorporate
- git slashes
- take issue, create branch/PR, address, push PR
- lint hooks with black/eslint/pyright/etc
- claude as linter for code review, etc.
- /permissions fetch,
- headless mode
- worktrees
- tmux
- hooks https://medium.com/@joe.njenga/use-claude-code-hooks-newest-feature-to-fully-automate-your-workflow-341b9400cfbe
    - more hook stuff: https://github.com/disler/claude-code-hooks-mastery?tab=readme-ov-file
- alias for dangerous, etc