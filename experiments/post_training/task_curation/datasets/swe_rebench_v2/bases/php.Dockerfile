# SWE-rebench V2.1 repair — php group base.
# Covers php_8.3.16 (377) and php:8.3.16 (47) rows — the same image, two declared spellings.
# Upstream anchor: the repo ships no php recipe; the declared base image php:8.3.16 is used
# directly (V2 row image_name ancestry) with the common upstream apt layer.
# Interpreter provisioning (repair, disclosed): apt python3 (bookworm ships 3.11).

FROM php:8.3.16@sha256:1625981bdf50872f0cf79f30aeb3095a5b13b112983dc82e2cb1f963eecadd74

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Disclosed repair normalization: apt runs non-interactively with --force-confdef/--force-confold
# (keeps the image's existing conffiles; an upgraded media-types prompted on mime.types and
# aborted non-interactive builds).

RUN apt-get update && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
    git curl wget unzip zip ca-certificates build-essential pkg-config cmake gnupg \
    libssl-dev libzip-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

RUN php -v | grep -q '^PHP 8.3' || { echo "php 8.3 required" >&2; exit 1; }

# Composer (the php rows' declared install/test commands run `composer install` and
# `./vendor/bin/phpunit`; without the composer binary the grading dies with exit 127 —
# the same missing-test-tooling class as pytest for python). Version PINNED to 2.2.26 —
# read from the upstream php image (docker.io/swerebenchv2/ampproject-amp-toolbox-php
# @61546a56 ships Composer 2.2.26, the LTS line the rows were built against).
RUN curl -fsSL https://getcomposer.org/download/2.2.26/composer.phar -o /usr/local/bin/composer \
    && chmod +x /usr/local/bin/composer \
    && composer --version 2>/dev/null | grep -q 'Composer version 2.2' \
    || { echo "composer 2.2.x required" >&2; exit 1; }

# Declared upstream-image php.ini/extension properties, evidenced by the diag-v21 captures:
#   - phar.readonly=0 (the upstream image's ini; box-561's 244 'Requires phar.readonly to be
#     set to 0' failures under the CLI default phar.readonly=1)
#   - ext-zip (amp-toolbox-513: 'requires PHP extension ext-zip * but it is missing from your
#     system' — composer install died before vendor/bin/phpunit existed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzip-dev \
    && docker-php-ext-install zip \
    && echo 'phar.readonly = 0' > /usr/local/etc/php/conf.d/zz-repair-phar-readonly.ini \
    && php -r 'var_export(ini_get("phar.readonly"));' | grep -q "'0'" \
    && php -m | grep -qi '^zip$' \
    || { echo "php base: zip ext or phar.readonly=0 missing" >&2; exit 1; } \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /logs /output && chmod 777 /logs /output
