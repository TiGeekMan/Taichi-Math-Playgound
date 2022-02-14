请首先安装依赖：

```
pip install -r requirements.txt
```

然后运行 `taichi_polyhedra_mirrors.py` (taichi 版本) 或者 `glsl_polyhedra_mirrors.py` (opengl 版本) 即可。


如果想要渲染其它的多面体，找到代码最下方的 `if __name__ == '__main__':` 的部分，入口函数调用为

``` python
generate_polytope_data(
    coxeter_diagram,
    trunc_type,
    extra_relations=(),
    snub=False,
    dual=False
):
```

各参数含义如下：

+ `coxeter_diagram` 是多面体对称群对应的 Coxeter-Dynkin 图。
+ `trunc_type` 设置多面体的截断类型。必须是三个非负实数且至少有一个不为 0。
+ `extra_relations` 用于指定星状多面体所需的额外生成关系 (目前的渲染代码不太支持这种多面体)。
+ `snub` 选择是否是手性多面体。
+ `dual` 选择是否渲染对偶多面体。

你可以修改这些参数来获得其它类型的多面体。