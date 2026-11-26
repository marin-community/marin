// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/moby/buildkit/frontend/dockerfile/instructions"
	"github.com/moby/buildkit/frontend/dockerfile/parser"
)

type Input struct {
	ID         string `json:"id"`
	Dockerfile string `json:"dockerfile"`
}

type Instruction struct {
	Attributes map[string]bool  `json:"attributes"`
	Command    string           `json:"command"`
	Original   string           `json:"original"`
	Flags      []string         `json:"flags"`
	Args       []string         `json:"args"`
	Start      int              `json:"start"`
	End        int              `json:"end"`
	Heredocs   []parser.Heredoc `json:"heredocs"`
}

func main() {
	scan := bufio.NewScanner(os.Stdin)
	scan.Buffer(make([]byte, 4096), 8*1024*1024)
	encoder := json.NewEncoder(os.Stdout)
	for scan.Scan() {
		var input Input
		if err := json.Unmarshal(scan.Bytes(), &input); err != nil {
			panic(err)
		}
		parsed, err := parser.Parse(strings.NewReader(input.Dockerfile))
		if err != nil {
			panic(fmt.Errorf("%s: %w", input.ID, err))
		}
		if _, _, err := instructions.Parse(parsed.AST, nil); err != nil {
			panic(fmt.Errorf("%s: %w", input.ID, err))
		}
		commands := []Instruction{}
		for _, node := range parsed.AST.Children {
			args := []string{}
			for next := node.Next; next != nil; next = next.Next {
				args = append(args, next.Value)
			}
			commands = append(commands, Instruction{node.Attributes, strings.ToLower(node.Value), node.Original, node.Flags, args, node.StartLine, node.EndLine, node.Heredocs})
		}
		if err := encoder.Encode(map[string]interface{}{"id": input.ID, "instructions": commands}); err != nil {
			panic(err)
		}
	}
	if err := scan.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
