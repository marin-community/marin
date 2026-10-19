// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// One persistent Bash PTY behind a line-framed virtio-serial control port.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#define PORT "/dev/vport0p1"
#define COMMAND_FILE "/tmp/harbor-pty-command"
#define LINE_CAPACITY 262144
#define CHUNK_BYTES 2048

static int port_fd = -1;
static int master_fd = -1;
static int status_fd = -1;
static pid_t shell_pid = -1;
static char active_id[32] = "";

static void write_all(int fd, const char *data, size_t size) {
    while (size) {
        ssize_t sent = write(fd, data, size);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) _exit(1);
        data += sent;
        size -= (size_t)sent;
    }
}

static void send_text(const char *text) { write_all(port_fd, text, strlen(text)); }

static int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static size_t decode_hex(char *text) {
    size_t length = strlen(text);
    if (length % 2) return 0;
    for (size_t index = 0; index < length / 2; index++) {
        int high = hex_digit(text[index * 2]);
        int low = hex_digit(text[index * 2 + 1]);
        if (high < 0 || low < 0) return 0;
        text[index] = (char)((high << 4) | low);
    }
    return length / 2;
}

static void send_output(const char *data, size_t length) {
    static const char digits[] = "0123456789abcdef";
    char line[4 + CHUNK_BYTES * 2 + 2];
    memcpy(line, "OUT|", 4);
    for (size_t index = 0; index < length; index++) {
        unsigned char byte = (unsigned char)data[index];
        line[4 + index * 2] = digits[byte >> 4];
        line[5 + index * 2] = digits[byte & 15];
    }
    line[4 + length * 2] = '\n';
    write_all(port_fd, line, 5 + length * 2);
}

static void drain_output(void) {
    for (;;) {
        struct pollfd fd = {master_fd, POLLIN, 0};
        if (poll(&fd, 1, 0) <= 0 || !(fd.revents & POLLIN)) return;
        char bytes[CHUNK_BYTES];
        ssize_t count = read(master_fd, bytes, sizeof(bytes));
        if (count <= 0) return;
        send_output(bytes, (size_t)count);
    }
}

static void start_shell(void) {
    int slave;
    int status_pipe[2];
    struct termios settings;
    if (openpty(&master_fd, &slave, NULL, NULL, NULL) < 0 || pipe(status_pipe) < 0) {
        perror("openpty or pipe");
        _exit(2);
    }
    if (tcgetattr(slave, &settings) == 0) {
        settings.c_lflag &= ~ECHO;
        tcsetattr(slave, TCSANOW, &settings);
    }
    struct winsize size = {.ws_row = 24, .ws_col = 80};
    ioctl(slave, TIOCSWINSZ, &size);
    shell_pid = fork();
    if (shell_pid < 0) _exit(2);
    if (shell_pid == 0) {
        close(master_fd);
        close(status_pipe[0]);
        if (setsid() < 0 || ioctl(slave, TIOCSCTTY, 0) < 0) _exit(2);
        tcsetpgrp(slave, getpgrp());
        dup2(slave, STDIN_FILENO);
        dup2(slave, STDOUT_FILENO);
        dup2(slave, STDERR_FILENO);
        dup2(status_pipe[1], 3);
        if (slave > 3) close(slave);
        if (status_pipe[1] > 3) close(status_pipe[1]);
        signal(SIGINT, SIG_DFL);
        signal(SIGQUIT, SIG_DFL);
        setenv("PS1", "", 1);
        setenv("HISTFILE", "/dev/null", 1);
        setenv("PROMPT_COMMAND",
               "__hb_status=$?; if [ -n \"${__hb_id:-}\" ]; then printf 'DONE|%s|%s\\n' \"$__hb_id\" \"$__hb_status\" >&3; unset __hb_id; fi",
               1);
        execl("/bin/bash", "bash", "--noprofile", "--norc", "-i", (char *)NULL);
        _exit(127);
    }
    close(slave);
    close(status_pipe[1]);
    status_fd = status_pipe[0];
    const char *ready = "bind 'set enable-bracketed-paste off'; printf 'READY\\n' >&3\n";
    write_all(master_fd, ready, strlen(ready));
}

static void reset_shell(void) {
    if (shell_pid > 0) {
        kill(-shell_pid, SIGKILL);
        waitpid(shell_pid, NULL, 0);
    }
    if (master_fd >= 0) close(master_fd);
    if (status_fd >= 0) close(status_fd);
    master_fd = -1;
    status_fd = -1;
    shell_pid = -1;
    if (active_id[0]) {
        send_text("RESET|shell_exited\n");
        active_id[0] = '\0';
    }
    start_shell();
}

static void handle_command(char *line) {
    if (!strcmp(line, "START")) {
        if (shell_pid < 0) start_shell();
        return;
    }
    if (!strncmp(line, "EXEC|", 5)) {
        char *id = line + 5;
        char *payload = strchr(id, '|');
        if (!payload || active_id[0] || strlen(id) >= LINE_CAPACITY - 5) {
            send_text("ERROR|busy_or_invalid\n");
            return;
        }
        *payload++ = '\0';
        if (!*id || strlen(id) >= sizeof(active_id)) {
            send_text("ERROR|invalid_id\n");
            return;
        }
        size_t length = decode_hex(payload);
        int command = open(COMMAND_FILE, O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (command < 0) _exit(2);
        write_all(command, payload, length);
        close(command);
        strcpy(active_id, id);
        char wrapper[160];
        int size = snprintf(wrapper, sizeof(wrapper), "__hb_id=%s; eval \"$(cat %s)\"\n", id, COMMAND_FILE);
        if (size <= 0 || (size_t)size >= sizeof(wrapper)) _exit(2);
        write_all(master_fd, wrapper, (size_t)size);
        return;
    }
    if (!strncmp(line, "INPUT|", 6)) {
        char *payload = line + 6;
        size_t length = decode_hex(payload);
        if (length) write_all(master_fd, payload, length);
        return;
    }
    if (!strcmp(line, "SIGNAL|INT")) {
        pid_t foreground = -1;
        if (ioctl(master_fd, TIOCGPGRP, &foreground) == 0 && foreground > 0) {
            kill(-foreground, SIGINT);
        } else {
            perror("TIOCGPGRP");
            write_all(master_fd, "\003", 1);
        }
        return;
    }
    send_text("ERROR|unknown_request\n");
}

int main(void) {
    for (int attempt = 0; attempt < 100 && port_fd < 0; attempt++) {
        port_fd = open(PORT, O_RDWR);
        if (port_fd < 0) usleep(100000);
    }
    if (port_fd < 0 || access("/bin/bash", X_OK) != 0) {
        perror("guest Bash service startup");
        _exit(3);
    }
    char input[LINE_CAPACITY];
    size_t input_length = 0;
    char status[256];
    size_t status_length = 0;
    for (;;) {
        struct pollfd fds[3] = {{port_fd, POLLIN, 0}, {master_fd, POLLIN, 0}, {status_fd, POLLIN, 0}};
        int ready = poll(fds, 3, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) _exit(2);
        if (fds[0].revents & POLLIN) {
            char bytes[4096];
            ssize_t count = read(port_fd, bytes, sizeof(bytes));
            if (count <= 0) _exit(0);
            for (ssize_t index = 0; index < count; index++) {
                char byte = bytes[index];
                if (byte == '\n') {
                    input[input_length] = '\0';
                    handle_command(input);
                    input_length = 0;
                } else if (input_length < sizeof(input) - 1) {
                    input[input_length++] = byte;
                } else {
                    _exit(2);
                }
            }
        }
        if (fds[1].revents & POLLIN) {
            char bytes[CHUNK_BYTES];
            ssize_t count = read(master_fd, bytes, sizeof(bytes));
            if (count > 0) send_output(bytes, (size_t)count);
        }
        if (fds[2].revents & POLLIN) {
            char bytes[128];
            ssize_t count = read(status_fd, bytes, sizeof(bytes));
            for (ssize_t index = 0; index < count; index++) {
                if (bytes[index] == '\n') {
                    status[status_length] = '\0';
                    if (!strcmp(status, "READY")) send_text("READY\n");
                    else if (!strncmp(status, "DONE|", 5)) {
                        drain_output();
                        send_text(status);
                        send_text("\n");
                        active_id[0] = '\0';
                    }
                    status_length = 0;
                } else if (status_length < sizeof(status) - 1) {
                    status[status_length++] = bytes[index];
                }
            }
        }
        if (shell_pid > 0 && waitpid(shell_pid, NULL, WNOHANG) == shell_pid) {
            shell_pid = -1;
            reset_shell();
        }
    }
}
