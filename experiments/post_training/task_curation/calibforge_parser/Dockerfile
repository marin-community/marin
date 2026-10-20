FROM golang:1.26.3-alpine
WORKDIR /src
COPY go.mod *.go ./
RUN go mod tidy && gofmt -w *.go && go build -o /atlas-parser . && go test ./...
FROM alpine:3.23
COPY --from=0 /atlas-parser /atlas-parser
ENTRYPOINT ["/atlas-parser"]
