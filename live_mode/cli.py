"""No-GPU interactive flow demonstration: python -m live_mode.cli."""
from __future__ import annotations
import argparse
from .stream import VectorCodec, LiveStream, DEFAULT_VALUES


def main() -> None:
    parser = argparse.ArgumentParser(description='Read old state while writing new experience')
    parser.add_argument('--lr', type=float, default=.85)
    args = parser.parse_args()
    flow = LiveStream(VectorCodec(DEFAULT_VALUES), learning_rate=args.lr)
    print('LiveMode | commands: see alice.color = blue | ask alice.color | stats | quit')
    print('Allowed values: ' + ', '.join(DEFAULT_VALUES))
    while True:
        try:
            line = input('live> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ('quit', 'exit'):
            return
        if line == 'stats':
            m = flow.memory
            print(f'events={m.events} writes={m.writes} memory={m.state_bytes} bytes')
        elif line.startswith('see ') and '=' in line:
            address, value = [part.strip() for part in line[4:].split('=', 1)]
            try:
                before, error = flow.observe(address, value)
                print(f'BEFORE read={before!r}; prediction error={error:.3f}; state updated')
            except (ValueError, IndexError) as e:
                print('error:', e)
        elif line.startswith('ask '):
            answer, score = flow.ask(line[4:].strip())
            print(f'READ {answer!r} (raw score={score:.3f}; not a probability)')
        else:
            print('Try: see alice.color = blue; ask alice.color; see alice.color = red')


if __name__ == '__main__':
    main()
