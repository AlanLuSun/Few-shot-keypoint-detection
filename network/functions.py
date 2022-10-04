from torch.autograd import Function


class ReverseLayerF(Function):
    '''
    Usage example:
    out = ReverseLayerF.apply(x, alpha)  # reverse_feature
    '''

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        # print(output)

        return output, None


