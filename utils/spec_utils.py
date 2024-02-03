import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.rigid_utils import exp_se3
from utils.quaternion_utils import init_predefined_omega
from utils.general_utils import linear_to_srgb
from utils.ref_utils import generate_ide_fn
import nvdiffrast.torch as dr


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi

        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)

    def forward(self, omega_o, a, la, mu):
        Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True))  # N, num_theta, num_phi, 1

        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (
                self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
        out = a * Smooth * torch.exp(exp_input)

        return out


class SGEnvmap(torch.nn.Module):
    def __init__(self, numLgtSGs=32, device='cuda'):
        super(SGEnvmap, self).__init__()

        self.lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
        self.lgtSGs.data[..., 3:4] *= 100.
        self.lgtSGs.data[..., -3:] = 0.
        self.lgtSGs.requires_grad = True

    def forward(self, viewdirs):
        lgtSGLobes = self.lgtSGs[..., :3] / (torch.norm(self.lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-7)
        lgtSGLambdas = torch.abs(self.lgtSGs[..., 3:4])  # sharpness
        lgtSGMus = torch.abs(self.lgtSGs[..., -3:])  # positive values
        pred_radiance = lgtSGMus[None] * torch.exp(
            lgtSGLambdas[None] * (torch.sum(viewdirs[:, None, :] * lgtSGLobes[None], dim=-1, keepdim=True) - 1.))
        reflection = torch.sum(pred_radiance, dim=1)

        return reflection


class ASGRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(ASGRender, self).__init__()

        self.num_theta = 4
        self.num_phi = 8
        self.ch_normal_dot_viewdir = 1
        self.in_mlpC = 2 * viewpe * 3 + 3 + self.num_theta * self.num_phi * 2 + self.ch_normal_dot_viewdir
        self.viewpe = viewpe
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, 'cuda')

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def safe_normalize(self, x, eps=1e-8):
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

    def forward(self, pts, viewdirs, features, normal):
        asg_params = features.view(-1, self.num_theta, self.num_phi, 4)  # [N, 8, 16, 4]
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        reflect_dir = self.safe_normalize(self.reflect(-viewdirs, normal))

        color_feature = self.ree_function(reflect_dir, a, la, mu)
        # color_feature = color_feature.view(color_feature.size(0), -1, 3)
        color_feature = color_feature.view(color_feature.size(0), -1)  # [N, 256]

        normal_dot_viewdir = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)  # [N, 1]
        indata = [color_feature, normal_dot_viewdir]
        if self.viewpe > -1:
            indata += [viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        # rgb = torch.sum(color_feature, dim=1)
        # rgb = torch.sigmoid(rgb)

        return rgb


class IdentityActivation(nn.Module):
    def forward(self, x): return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light = max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))


def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
                   exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation == 'none':
        activation = IdentityActivation()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module = nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module


class AppShadingNetwork(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
    }

    def __init__(self):
        super().__init__()
        self.cfg = {**self.default_cfg}
        feats_dim = 256

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8
        # self.asg_hidden = self.num_theta * self.num_phi * 5
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # # inner lights are indirect lights
        # self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        # nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        # self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        # nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        self.render_module = ASGRender(self.asg_hidden, 2, 2, 128)

    def predict_specular_lights(self, points, reflective, roughness, step):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        direct_light = self.outer_light(ref_roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
        occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, normals):
        roughness = torch.ones([normals.shape[0], 1]).cuda()
        ref = self.sph_enc(normals, roughness)
        light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, albedo, roughness, metallic, inter_results=False):
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        # # diffuse light
        # diffuse_albedo = (1 - metallic) * albedo
        # diffuse_light = self.predict_diffuse_lights(normals)
        # diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        # specular_light, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, reflective,
        #                                                                                      roughness, step)
        specular_light = self.render_module(points, -view_dirs, feature_vectors, normals)

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
                               boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        specular_color = specular_ref * specular_light

        # integrated together
        # color = diffuse_color + specular_color

        # gamma correction
        # diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        # color = linear_to_srgb(color)
        # color = torch.clamp(color, min=0.0, max=1.0)

        if inter_results:
            intermediate_results = {
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color, min=0.0, max=1.0),

                # 'diffuse_albedo': diffuse_albedo,
                # 'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                # 'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'metallic': metallic,
                'roughness': roughness,
            }
            return specular_color, intermediate_results
        else:
            return specular_color


class SpecularNetwork(nn.Module):
    def __init__(self, D=4, W=128, input_ch=3, output_ch=59, view_multires=4, multires=4):
        super(SpecularNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.view_multires = view_multires
        self.skips = [D // 2]

        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8
        # self.asg_hidden = self.num_theta * self.num_phi * 5
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # self.embed_view_fn, view_input_ch = get_embedder(view_multires, 3)
        # self.embed_fn, xyz_input_ch = get_embedder(multires, self.asg_feature)
        # self.input_ch = xyz_input_ch

        # self.linear = nn.ModuleList(
        #     [nn.Linear(self.input_ch, W)] + [
        #         nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
        #         for i in range(D - 1)]
        # )
        # self.env_module = SGEnvmap()

        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)

        self.render_module = ASGRender(self.asg_hidden, 2, 2, 128)

    def forward(self, x, view, normal):
        # v_emb = self.embed_view_fn(view)
        # x_emb = self.embed_fn(x)
        # h = torch.cat([x_emb, v_emb], dim=-1)
        # h = x
        # for i, l in enumerate(self.linear):
        #     h = self.linear[i](h)
        #     h = F.relu(h)
        #     if i in self.skips:
        #         h = torch.cat([x_emb, h], -1)

        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal)
        # reflect = self.env_module(reflect_dir)

        return spec
