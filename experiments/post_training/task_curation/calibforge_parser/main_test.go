package main

import (
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

func TestWirePreservesShellAndExecForms(t *testing.T) {
	input := Input{ID: "forms", Dockerfile: "FROM ubuntu:24.04\nRUN echo $HOME\nRUN [\"echo\", \"$HOME\"]\nCOPY [\"file with spaces\", \"/app/\"]\n"}
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	command := exec.Command("/atlas-parser")
	command.Stdin = strings.NewReader(string(data) + "\n")
	output, err := command.Output()
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Instructions []Instruction `json:"instructions"`
	}
	if err := json.Unmarshal(output, &result); err != nil {
		t.Fatal(err)
	}
	if result.Instructions[1].Attributes["json"] || !result.Instructions[2].Attributes["json"] {
		t.Fatal("shell and exec RUN forms were collapsed")
	}
	if result.Instructions[3].Args[0] != "file with spaces" || !result.Instructions[3].Attributes["json"] {
		t.Fatal("JSON COPY source boundary was lost")
	}
}
