[How to incrementally sample without replacement?](https://stackoverflow.com/questions/18921302/how-to-incrementally-sample-without-replacement)


```python
import random

def sample_gen(n, forbid):
    state = dict()
    track = dict()
    for (i, o) in enumerate(forbid):
        x = track.get(o, o)
        t = state.get(n-i-1, n-i-1)
        state[x] = t
        track[t] = x
        state.pop(n-i-1, None)
        track.pop(o, None)
    del track
    for remaining in xrange(n-len(forbid), 0, -1):
        i = random.randrange(remaining)
        yield state.get(i, i)
        state[i] = state.get(remaining - 1, remaining - 1)
        state.pop(remaining - 1, None)
```

## follow up

- very long array can not fit into memory
