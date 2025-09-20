package main

import (
	"os"
	"testing"
)

func TestMain(t *testing.T) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = os.Args[:1]
	main()
}
