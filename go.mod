module github.com/luxfi/lattice/v7

go 1.26.1

require (
	github.com/ALTree/bigfloat v0.2.0
	github.com/google/go-cmp v0.7.0
	github.com/luxfi/math v2.0.0+incompatible
	github.com/stretchr/testify v1.11.1
	golang.org/x/crypto v0.49.0
	golang.org/x/exp v0.0.0-20260312153236-7ab1446f8b90
)

// LP-107 Phase 3: math has not been published yet. Point at the local
// canonical-relocation branch on disk; the user replaces this with a
// proper version pin after pushing.
replace github.com/luxfi/math => /Users/z/work/lux/math/.claude/worktrees/lp-107-phase3

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/rogpeppe/go-internal v1.14.1 // indirect
	golang.org/x/sys v0.42.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
